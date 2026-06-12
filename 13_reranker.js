import { Document } from "@langchain/core/documents";
import { OllamaEmbeddings } from "@langchain/ollama";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";

console.log("=== Reranking Demo ===\n");

const chunks = [
  "Tesla reported record quarterly revenue of $25.2 billion in Q3 2024.",
  "Tesla Cybertruck production begins in 2024.",
  "Tesla announced plans to expand Gigafactory production capacity.",
  "Tesla stock price reached new highs after earnings.",
  "Tesla continues to lead the electric vehicle market.",
  "Microsoft acquired GitHub for $7.5 billion in 2018.",
  "Google announced Gemini AI model.",
  "NVIDIA announced next-generation Blackwell architecture.",
  "The Tesla coil was invented by Nikola Tesla in 1891.",
  "Financial markets showed volatility during earnings season.",
];

const documents = chunks.map(
  (chunk, index) =>
    new Document({
      pageContent: chunk,
      metadata: { id: index },
    })
);

async function main() {
  const embeddings = new OllamaEmbeddings({
    model: "nomic-embed-text",
    baseUrl: "http://localhost:11434",
  });

  const vectorStore = await MemoryVectorStore.fromDocuments(
    documents,
    embeddings
  );

  const query = "Tesla financial performance and production updates";

  console.log("STEP 1: Retrieval");
  console.log("-".repeat(40));

  const retrievedDocs = await vectorStore.similaritySearch(query, 8);

  retrievedDocs.forEach((doc, i) => {
    console.log(`${i + 1}. ${doc.pageContent}`);
  });

  console.log("\nSTEP 2: Manual Reranking");
  console.log("-".repeat(40));

  const reranked = retrievedDocs.sort((a, b) => {
    const scoreA =
      (a.pageContent.toLowerCase().includes("revenue") ? 3 : 0) +
      (a.pageContent.toLowerCase().includes("production") ? 2 : 0) +
      (a.pageContent.toLowerCase().includes("earnings") ? 1 : 0);

    const scoreB =
      (b.pageContent.toLowerCase().includes("revenue") ? 3 : 0) +
      (b.pageContent.toLowerCase().includes("production") ? 2 : 0) +
      (b.pageContent.toLowerCase().includes("earnings") ? 1 : 0);

    return scoreB - scoreA;
  });

  reranked.forEach((doc, i) => {
    console.log(`${i + 1}. ${doc.pageContent}`);
  });

  console.log("\n✅ Reranking Completed");
}

main().catch(console.error);