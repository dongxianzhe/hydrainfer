from typing import Optional
import requests
import json
import urllib.parse
from hydrainfer.utils.logger import getLogger
logger = getLogger(__name__)

class HTTPStoragePlugin:
    def __init__(self, metadata_uri: str):
        self.metadata_uri = metadata_uri

    def _encode_url(self, key: str) -> str:
        encoded_key = urllib.parse.quote(key)
        return f"{self.metadata_uri}?key={encoded_key}"

    def get(self, key: str) -> tuple[bool, Optional[dict]]:
        url = self._encode_url(key)
        try:
            response = requests.get(url, timeout=3)
            if response.status_code != 200:
                logger.error(f"Unexpected response code: {response.status_code}")
                logger.error(f"Response body: {response.text}")
                return False, None
            # Try to parse the response JSON
            try:
                value = response.json()
                return True, value
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON from response: {response.text}")
                return False, None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error in GET request: {e}")
            return False, None

    def set(self, key: str, value: dict) -> bool:
        # value is json serializable
        url = self._encode_url(key)
        try:
            headers = {'Content-Type': 'application/json'}
            response = requests.put(url, json=value, headers=headers, timeout=3)
            if response.status_code != 200:
                logger.error(f"Unexpected response code: {response.status_code}")
                logger.error(f"Response body: {response.text}")
                return False
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Error in PUT request: {e}")
            return False

    def remove(self, key: str) -> bool:
        url = self._encode_url(key)
        try:
            response = requests.delete(url, timeout=3)
            if response.status_code != 200:
                logger.error(f"Unexpected response code: {response.status_code}")
                logger.error(f"Response body: {response.text}")
                return False
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Error in DELETE request: {e}")
            return False
