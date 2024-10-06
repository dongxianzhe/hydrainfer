from dxz.engine.llm_engine import LLMEngine

if __name__ == '__main__':
    llm_engine = LLMEngine()
    results = llm_engine.generate(["Hello"])
    for result in results:
        print(result)