import fs from "fs";
import path from "path";

import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OllamaEmbeddings } from "@langchain/ollama";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";

async function main() {
  console.log("=== Retrieval Methods using Node.js + Ollama ===\n");

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

  const embeddings = new OllamaEmbeddings({
    model: "nomic-embed-text",
    baseUrl: "http://localhost:11434",
  });

  const vectorStore = await MemoryVectorStore.fromDocuments(
    chunks.slice(0, 120),
    embeddings
  );

  const query = "How much did Microsoft pay to acquire GitHub?";

  console.log(`Query: ${query}\n`);

  console.log("=== METHOD 1: Similarity Search (k=3) ===");

  const similarityDocs = await vectorStore.similaritySearch(query, 3);

  console.log(`Retrieved ${similarityDocs.length} documents:\n`);

  similarityDocs.forEach((doc, index) => {
    console.log(`Document ${index + 1}:`);
    console.log(doc.pageContent);
    console.log("\n" + "-".repeat(60) + "\n");
  });

  console.log("=== METHOD 2: Similarity Search With Scores ===");

  const scoredDocs = await vectorStore.similaritySearchWithScore(query, 3);

  scoredDocs.forEach(([doc, score], index) => {
    console.log(`Document ${index + 1}`);
    console.log(`Score: ${score}`);
    console.log(doc.pageContent);
    console.log("\n" + "-".repeat(60) + "\n");
  });

  console.log("=== METHOD 3: MMR Style Search (Manual Diversity Demo) ===");

  const moreDocs = await vectorStore.similaritySearch(query, 8);

  const uniqueDocs = [];
  const seen = new Set();

  for (const doc of moreDocs) {
    const key = doc.pageContent.slice(0, 120);

    if (!seen.has(key)) {
      seen.add(key);
      uniqueDocs.push(doc);
    }

    if (uniqueDocs.length === 3) break;
  }

  uniqueDocs.forEach((doc, index) => {
    console.log(`Diverse Document ${index + 1}:`);
    console.log(doc.pageContent);
    console.log("\n" + "-".repeat(60) + "\n");
  });

  console.log("Done! Retrieval methods tested successfully.");
}

main().catch((error) => {
  console.error("❌ Error:", error.message);
});