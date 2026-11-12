
### 启动命令
- dev： gunicorn start_platform_rag:app -b 0.0.0.0:9900 -c gunicorn_conf.py -e env=dev -e log-dir=./logs -w 1

- test： gunicorn start_platform_rag:app -b 0.0.0.0:80 -c gunicorn_conf.py -e service_name=nlp-lab -e zone=defaultZone -e env="${PROFILE}" -e log-dir="${LOGPATH}" -w 6

- test： gunicorn start_platform_rag:app -b 0.0.0.0:80 -c gunicorn_conf.py -e service_name=nlp-lab -e zone=defaultZone -w 6

- prod: gunicorn start_platform_rag:app -b 0.0.0.0:80 -c gunicorn_conf.py -e service_name=nlp-lab -e zone=defaultZone -e env="${PROFILE}" -e log-dir="${LOGPATH}" -w 6




