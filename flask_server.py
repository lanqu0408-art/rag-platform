import os
import logging

import argparse
logging.basicConfig(level=logging.INFO)
os.environ['env'] = 'dev'

from start_platform_rag import app


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='NLP Server'
    )
    parser.add_argument(
        '-host', type=str, default='0.0.0.0',
        help='Server host address'
    )
    parser.add_argument(
        '-port', type=int, default=9901,
        help='Server port number'
    )
    parser.add_argument(
        '-debug', action='store_true',
        help='Enable debug mode'
    )
    args = parser.parse_args()
    return args


args = parse_args()

if __name__ == '__main__':
    app.run(host=args.host, port=args.port, debug=args.debug)

# python flask_server.py -host 0.0.0.0 -port 9901
# uv run flask_server.py  -host 0.0.0.0 -port 9901