# RecursiveAI.OneOcr.Llm

Opt-in LLM vision for `RecursiveAI.OneOcr`: Anthropic, OpenAI, OpenAI-compatible servers (Ollama, llama.cpp, vLLM, LM Studio), Gemini and Azure OpenAI over a plain `HttpClient`. This is the only OneOcr package that contains networking code; nothing leaves the machine unless you configure a provider, and local methods never touch the network.

```sh
dotnet add package RecursiveAI.OneOcr.Llm
```

Every value a model returns is checked against the words the local reader found on the page and comes back as `Grounded`, `GroundedNormalized`, `Derived` or `Ungrounded`, with boxes where it has them. The package also provides cost estimates and caps, routing between the local models and the LLM, an `Offline` switch, a redaction hook and structured output from `System.Text.Json` source-generated types. API keys are passed explicitly; nothing is read from the environment unless you ask.

https://github.com/recursive-ai-oss/oneocr-csharp
