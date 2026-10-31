### Crucial Hardware Note for Apple M1 (8 GB RAM)
  
  Because your machine has 8 GB Unified Memory, running Qwen3-Embedding-8B in standard FP16 (~16 GB) would cause
  severe disk swapping. In our Backend LLD, we should specifically design for your M1:                          
                                                                                                                
  1. Qwen3-Embedding-8B in 4-bit Quantization (Q4_K_M GGUF via Metal / Ollama or llama-cpp-python): Shrinks     
  memory footprint from 16 GB down to ~4.6 GB while keeping Matryoshka (1024d) slicing and L2 normalization in  
  pure NumPy.                                                                                                   
  2. Sequential / On-Demand Model Lifecycle: Unload Granite-Docling (258M) after document parsing completes, and
  coordinate memory between Qwen3-Embedding-8B and the generation LLM so total RAM stays well under 8 GB.       


