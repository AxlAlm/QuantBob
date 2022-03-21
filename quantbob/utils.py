
import hashlib


def dict2uid(input_dict:dict) -> str:
    hashlib.sha256(str(input_dict).encode()).hexdigest()