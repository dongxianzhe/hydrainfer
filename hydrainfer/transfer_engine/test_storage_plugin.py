from hydrainfer.transfer_engine.storage_plugin import HTTPStoragePlugin

if __name__ == '__main__':
    storage = HTTPStoragePlugin('http://localhost:8082/metadata')
    print(storage.get(key="user:123"))
    print(storage.set(key="user:123", value={"name": "John Doe", "age": 30}))
    print(storage.get(key="user:123"))
    print(storage.remove(key="user:123"))
    print(storage.get(key="user:123"))