from flask import Flask, request, jsonify
import sys
import json
import threading
import random
import requests
import time
from datetime import datetime, timedelta

app = Flask(__name__)

heartbeat_lock = threading.Lock()
election_timeout = random.uniform(0.25, 0.5)
just_joined = True
leader_curr = None
leader_ip = None
last_heartbeat_time = None
last_applied_index = -1
commit_index=-1
status = {"role": "Follower", "term": 0, "votedFor": None}
configuration = {}
node_configuration = {}
state = {}
other_nodes = []
local_log = []

def become_candidate():
    global status, leader_curr, election_timeout
    status["role"] = "candidate"
    status["term"] += 1
    status["votedFor"] = node_configuration['port']
    votes_received = 1 
    leader_curr = None

    print(f"Node at port {node_configuration['port']} becoming candidate for term {status['term']}.")

    for peer in other_nodes:
        try:
            response = requests.get(f"{peer}/vote", params={"term": status["term"], "candidatePort": node_configuration['port'], "candidateLog": local_log}, timeout=1)
            if response.status_code == 200 and response.json().get('giveVote'):
                votes_received += 1
        except Exception:
            continue
    if votes_received > len(other_nodes) / 2:
        become_leader()
    else:
        revert(status["term"])

def revert(new_term):
    global status, leader_curr, last_heartbeat_time
    status["role"] = "Follower"
    status["term"] = new_term
    reset_election_timeout()
    print(f"Node at port {node_configuration['port']} is now a Follower.")

def become_leader():
    global status, leader_curr, leader_ip
    status["role"] = "Leader"
    leader_curr = node_configuration['port']
    leader_ip = node_configuration['ip'] 
    print(f"Node at IP {leader_ip} and port {leader_curr} is now the leader for term {status['term']}.")
    send_heartbeat()

def send_heartbeat():
    global status, leader_curr
    while status["role"] == "Leader":
        for peer in other_nodes:
            append_entries(peer, status["term"], leader_curr, leader_ip, [], commitIndex=commit_index)
        time.sleep(0.2)


def append_entries(peer, term, leader_id, leader_ip, entries, commit=False, commitIndex=None):
    payload = {
        "term": term,
        "leaderPort": leader_id,
        "entries": entries,
        "commitIndex": commitIndex,
        "leaderIp": leader_ip
    }
    try:
        response = requests.post(f"{peer}/append_entries", json=payload, timeout=1)
        return response.json() 
    except requests.RequestException:
        pass

@app.route('/vote', methods=['GET'])
def request_vote():
    global status
    candidate_term = request.args.get('term', type=int)
    candidate_port = request.args.get('candidatePort', type=int)

    if candidate_term > status["term"]:
        status["term"] = candidate_term
        status["votedFor"] = None

    if ((status["votedFor"] == candidate_port and candidate_term == status["term"]) or (status["votedFor"] is None)):
        status["votedFor"] = candidate_port
        reset_election_timeout()
        return jsonify(giveVote=True)
    return jsonify(giveVote=False)

def apply_log_entry(entry, state):
    if entry['action'] == 'create_topic':
        state[entry['topic']] = []
    elif entry['action'] == 'add_message':
        state[entry['topic']].append(entry['message'])
    elif entry['action'] == 'pop_message':
        state[entry['topic']].pop(0)
    
@app.route('/append_entries', methods=['POST'])
def handle_append_entries():
    global status, leader_curr, last_heartbeat_time, local_log, state, last_applied_index, commit_index
    data = request.json
    term = data['term']
    entries = data['entries']
    leader_id = data['leaderPort']
    leader_addr = data['leaderIp']
    leader_commit_index = data.get('commitIndex', None)

    if term >= status["term"]:
        status["term"] = term
        status["role"] = "Follower"
        leader_curr = leader_id
        local_log.extend(elem for elem in entries if elem not in local_log)
        reset_election_timeout()

        if len(local_log) < leader_commit_index + 1:
            response = requests.post(f"http://{leader_addr}:{leader_id}/request_entries", json={'index': len(local_log)})
            if response.status_code == 200:
                entries_to_add = response.json().get('entries', [])
                local_log.extend(entries_to_add)
                for i in range(last_applied_index + 1, min(leader_commit_index + 1, len(local_log))):
                    entry = local_log[i]
                    apply_log_entry(entry, state)
                    last_applied_index = i
                commit_index = min(leader_commit_index, last_applied_index)

        if leader_commit_index is not None and leader_commit_index < len(local_log):
            for i in range(last_applied_index + 1, leader_commit_index + 1):
                entry = local_log[i]
                apply_log_entry(entry, state)
                last_applied_index = i
        return jsonify(success=True)
    else:
        return jsonify(success=False)

@app.route('/request_entries', methods=['POST'])
def handle_request_entries():
    global local_log
    data = request.json
    index = data['index']

    if 0 <= index < len(local_log):
        missing_entries = local_log[index:]
        return jsonify(entries=missing_entries)
    else:
        return jsonify(error="Index out of bounds"), 400

def reset_election_timeout():
    global election_timeout, last_heartbeat_time
    election_timeout = random.uniform(0.25, 0.5)
    last_heartbeat_time = datetime.now()

def check_election_timeout():
    global last_heartbeat_time, election_timeout
    if last_heartbeat_time is None:
        reset_election_timeout()
    while True:
        time.sleep(0.15) 
        with heartbeat_lock:
            if status["role"] != "Leader":
                if datetime.now() - last_heartbeat_time > timedelta(seconds=election_timeout):
                    print(f"Election timeout exceeded at node {node_configuration['port']}. Becoming a candidate.")
                    become_candidate()
                    reset_election_timeout()

@app.route('/topic', methods=['PUT'])
def create_topic():
    global commit_index
    global local_log
    if status['role'] != 'Leader':
        return jsonify({"error": "This node is not the leader. Please send your request to the leader node."}), 403

    data = request.json
    topic = data.get('topic')
    if topic is None or topic in state:
        return jsonify(success=False), 400

    log_entry = {'action': 'create_topic', 'topic': topic, 'term': status["term"]}
    local_log.append(log_entry)

    successful_appends = 1 
    for peer in other_nodes:
        response = append_entries(peer, status["term"], node_configuration['port'], node_configuration['ip'], local_log, commitIndex=commit_index)
        if response and response.get('success'):
            successful_appends += 1

    if successful_appends > len(other_nodes) / 2:
        commit_index = len(local_log) - 1
        state[topic] = [] # Committing locally on the leader
        # Tell other nodes to commit the action
        for peer in other_nodes:
            append_entries(peer, status["term"], node_configuration['port'], node_configuration['ip'], [], commit=True, commitIndex=len(local_log) - 1)
        return jsonify(success=True)
    else:
        return jsonify(success=False), 500

@app.route('/topic', methods=['GET'])
def get_topics():
    if status['role'] != 'Leader':
        return jsonify({"error": "This node is not the leader. Please send your request to the leader node."}), 403
    return jsonify(success=True, topics=list(state.keys()))

@app.route('/message', methods=['PUT'])
def add_message():
    global commit_index
    global local_log
    if status['role'] != 'Leader':
        return jsonify({"error": "This node is not the leader. Please send your request to the leader node."}), 403
    data = request.json
    topic = data.get('topic')
    message = data.get('message')
    if topic is None or message is None or topic not in state:
        return jsonify(success=False), 400
    
    log_entry = {'action': 'add_message', 'topic': topic, 'message': message, 'term': status["term"]}
    local_log.append(log_entry) 

    successful_appends = 1
    for peer in other_nodes:
        response = append_entries(peer, status["term"], node_configuration['port'], node_configuration['ip'], local_log, commitIndex=commit_index)
        if response and response.get('success'):
            successful_appends += 1

    if successful_appends > len(other_nodes) / 2:
        commit_index = len(local_log) - 1
        state[topic].append(message)
        for peer in other_nodes:
            append_entries(peer, status["term"], node_configuration['port'], node_configuration['ip'], [], commit=True, commitIndex=len(local_log) - 1)
        return jsonify(success=True)
    else:
        return jsonify(success=False), 500

@app.route('/message/<topic>', methods=['GET'])
def get_message(topic):
    global commit_index
    if status['role'] != 'Leader':
        return jsonify({"error": "This node is not the leader. Please send your request to the leader node."}), 403
    if topic not in state or len(state[topic]) == 0:
        return jsonify(success=False), 404
    
    log_entry = {'action': 'pop_message', 'topic': topic, 'term': status["term"]}
    local_log.append(log_entry)

    successful_appends = 1
    for peer in other_nodes:
        response = append_entries(peer, status["term"], node_configuration['port'], node_configuration['ip'], local_log, commitIndex=commit_index)
        if response and response.get('success'):
            successful_appends += 1
    
    if successful_appends > len(other_nodes) / 2:
        commit_index = len(local_log) - 1
        message = state[topic].pop(0)
        for peer in other_nodes:
            append_entries(peer, status["term"], node_configuration['port'], node_configuration['ip'], [], commit=True, commitIndex=len(local_log) - 1)

    return jsonify(success=True, message=message)


@app.route('/status', methods=['GET'])
def get_status():
    return jsonify(role=status["role"], term=status["term"], leader=leader_curr)

@app.route('/log', methods=['GET'])
def get_log():
    return jsonify(log=local_log)

@app.route('/contents', methods=['GET'])
def get_contents():
    return jsonify(contents=state)

if __name__ == '__main__': 
    filepath = sys.argv[1]
    idx_node = int(sys.argv[2])
    with open(filepath) as configuration_file:
        configuration = json.load(configuration_file)
        node_configuration = configuration['addresses'][idx_node]
        other_nodes = [f"http://{node['ip']}:{node['port']}" for i, node in enumerate(configuration['addresses']) if i != idx_node]

    threading.Thread(target=check_election_timeout).start()

    app.run(debug=False, host='0.0.0.0', port=node_configuration['port'])