# ELK Stack Lab - Native Ubuntu
Elasticsearch, Kibana, Logstash 8.12.0 on Ubuntu 26.04 LTS (x86_64), headless over SSH.
## Deviations from the reference WSL guide
1. No separate Java install - ES and Logstash 8.x bundle JDK 21 and read ES_JAVA_HOME / LS_JAVA_HOME, not JAVA_HOME.
2. Extracted to /opt/elk, not /home. /home is the parent of user home directories; /opt is for third-party software.
3. No --allow-root. Elasticsearch refuses to run as root by design; chown -R the install dir to the running user instead.
4. No /mnt/c/ paths - that is the WSL bridge to Windows and does not exist natively.
## Host preparation
sudo sysctl -w vm.max_map_count=262144 - ES memory-maps index files and will not start below this. Ubuntu default is 65530. Persisted in /etc/sysctl.conf.
JVM heap capped at 1g per component in config/jvm.options.d/heap.options, since three JVMs share one laptop.
## Headless access
Kibana binds to localhost:5601 only. Reached by SSH tunnel rather than exposing the port:
ssh -L 5601:localhost:5601 -L 9200:localhost:9200 user@host
## Pipeline (config/auth.conf)
Reads /var/log/auth.log, indexes to ssh-auth.
Timestamp format: Ubuntu 26.04 writes RFC3339 (2026-11-02T00:48:26.521117+00:00), not legacy syslog. SYSLOGTIMESTAMP does not match; TIMESTAMP_ISO8601 is required. Patterns from older tutorials fail silently and tag every event _grokparsefailure.
Two-stage grok: first splits the syslog envelope (timestamp, host, program, optional PID); second extracts SSH login fields. Non-SSH lines are tagged not_ssh_auth, distinguishing "not an SSH event" from "pattern is broken".
Field naming: parsed hostname stored as log_host, not host - under ECS v8 host is reserved as an object and a string there causes a mapping conflict.
Credentials: password held in the Logstash keystore as ${ES_PW}, not in the config file.
bin/logstash-keystore create
echo -n 'password' | bin/logstash-keystore add ES_PW --stdin
## Running
cd /opt/elk/elasticsearch-8.12.0 && ./bin/elasticsearch
cd /opt/elk/kibana-8.12.0 && ./bin/kibana
cd /opt/elk/logstash-8.12.0 && bin/logstash -f config/auth.conf
Elasticsearch must be up first; Kibana will not start without it.
## Notes
Index health is yellow on a single node: one replica is configured but ES will not place a replica on the same node as its primary. All primaries assigned - expected, not a fault.
Dynamic mapping makes every string a text field with a .keyword subfield. Aggregations and visualizations must use .keyword.
