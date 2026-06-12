import fs from "fs";
import path from "path";

import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OllamaEmbeddings } from "@langchain/ollama";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";

async function main() {
  console.log("=== Multi Query Retrieval using Ollama ===\n");

  const filePath = path.join("docs", "Tesla.txt");
  const content = fs.readFileSync(filePath, "utf8");

  const docs = [
    new Document({
      pageContent: content,
      metadata: { source: filePath },
    }),
  ];

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 700,
    chunkOverlap: 100,
  });

  const chunks = await splitter.splitDocuments(docs);

  const embeddings = new OllamaEmbeddings({
    model: "nomic-embed-text",
    baseUrl: "http://localhost:11434",
  });

  const vectorStore = await MemoryVectorStore.fromDocuments(
    chunks.slice(0, 120),
    embeddings
  );

  const originalQuery = "How does Tesla make money?";

  const queryVariations = [
    "What are Tesla's main sources of revenue?",
    "How does Tesla generate income from its business?",
    "What products and services contribute to Tesla's earnings?",
  ];

  console.log(`Original Query: ${originalQuery}\n`);

  console.log("Generated Query Variations:");
  queryVariations.forEach((query, index) => {
    console.log(`${index + 1}. ${query}`);
  });

  console.log("\n" + "=".repeat(60));

  for (let i = 0; i < queryVariations.length; i++) {
    const query = queryVariations[i];

    console.log(`\n=== RESULTS FOR QUERY ${i + 1}: ${query} ===`);

    const results = await vectorStore.similaritySearch(query, 5);

    console.log(`Retrieved ${results.length} documents:\n`);

    results.forEach((doc, index) => {
      console.log(`Document ${index + 1}:`);
      console.log(doc.pageContent.slice(0, 250));
      console.log("...\n");
    });

    console.log("-".repeat(50));
  }

  console.log("\n✅ Multi-Query Retrieval Complete!");
}

main().catch((error) => {
  console.error("❌ Error:", error.message);
});