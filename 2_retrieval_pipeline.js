import fs from "fs";
import path from "path";

import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OllamaEmbeddings } from "@langchain/ollama";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";

async function main() {
  console.log("=== Retrieval Pipeline using Node.js + Ollama ===\n");

  const filePath = path.join("docs", "Microsoft.txt");
  const content = fs.readFileSync(filePath, "utf-8");

  const documents = [
    new Document({
      pageContent: content,
      metadata: { source: filePath },
    }),
  ];

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 700,
    chunkOverlap: 100,
  });

  const chunks = await splitter.splitDocuments(documents);

  console.log(`Total Microsoft chunks: ${chunks.length}`);

  const embeddings = new OllamaEmbeddings({
    model: "nomic-embed-text",
    baseUrl: "http://localhost:11434",
  });

  const vectorStore = await MemoryVectorStore.fromDocuments(
    chunks.slice(0, 120),
    embeddings
  );

  const query = "How much did Microsoft pay to acquire GitHub?";

  const results = await vectorStore.similaritySearch(query, 5);

  console.log(`\nUser Query: ${query}`);
  console.log("\n--- Context ---\n");

  results.forEach((doc, index) => {
    console.log(`Document ${index + 1}:`);
    console.log(doc.pageContent);
    console.log("\n" + "-".repeat(50) + "\n");
  });
}

main().catch((error) => {
  console.error("❌ Error:", error.message);
});