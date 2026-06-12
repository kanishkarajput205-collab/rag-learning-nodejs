import fs from "fs";
import path from "path";
import { Document } from "@langchain/core/documents";
import { CharacterTextSplitter } from "@langchain/textsplitters";
import { OllamaEmbeddings } from "@langchain/ollama";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";

function loadDocuments(docsPath = "docs") {
  console.log(`Loading documents from ${docsPath}...`);

  if (!fs.existsSync(docsPath)) {
    throw new Error(`The directory ${docsPath} does not exist.`);
  }

  const files = fs.readdirSync(docsPath).filter((file) => file.endsWith(".txt"));

  if (files.length === 0) {
    throw new Error(`No .txt files found in ${docsPath}.`);
  }

  return files.map((file) => {
    const filePath = path.join(docsPath, file);
    const content = fs.readFileSync(filePath, "utf-8");

    return new Document({
      pageContent: content,
      metadata: { source: filePath },
    });
  });
}

async function splitDocuments(documents) {
  console.log("Splitting documents into chunks...");

  const textSplitter = new CharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 100,
  });

  const chunks = await textSplitter.splitDocuments(documents);

  console.log(`Total chunks created: ${chunks.length}`);
  console.log("Using first 20 chunks for testing...");

  return chunks.slice(0, 20);
}

async function createVectorStore(chunks) {
  console.log("Creating embeddings using Ollama...");

  const embeddingModel = new OllamaEmbeddings({
    model: "nomic-embed-text",
    baseUrl: "http://localhost:11434",
  });

  console.log("Creating memory vector store...");

  const vectorStore = await MemoryVectorStore.fromDocuments(
    chunks,
    embeddingModel
  );

  console.log("Memory vector store created successfully.");

  return vectorStore;
}

async function main() {
  console.log("=== RAG Ingestion Pipeline using Node.js + Ollama ===\n");

  const documents = loadDocuments("docs");
  console.log(`Documents loaded: ${documents.length}`);

  const chunks = await splitDocuments(documents);

  await createVectorStore(chunks);

  console.log("\n✅ Ingestion completed successfully using Ollama.");
}

main().catch((error) => {
  console.error("❌ Error:", error.message);
});