import fs from "fs";
import path from "path";

import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OllamaEmbeddings } from "@langchain/ollama";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";

function reciprocalRankFusion(chunkLists, k = 60) {
  console.log("\n" + "=".repeat(60));
  console.log("APPLYING RECIPROCAL RANK FUSION");
  console.log("=".repeat(60));

  const scores = new Map();
  const docsMap = new Map();

  chunkLists.forEach((chunks, queryIndex) => {
    console.log(`\nProcessing Query ${queryIndex + 1} results:`);

    chunks.forEach((doc, positionIndex) => {
      const position = positionIndex + 1;
      const key = doc.pageContent;
      const score = 1 / (k + position);

      scores.set(key, (scores.get(key) || 0) + score);
      docsMap.set(key, doc);

      console.log(
        `Position ${position}: +${score.toFixed(4)} | Total: ${scores
          .get(key)
          .toFixed(4)}`
      );
    });
  });

  return Array.from(scores.entries())
    .map(([content, score]) => ({
      doc: docsMap.get(content),
      score,
    }))
    .sort((a, b) => b.score - a.score);
}

async function main() {
  console.log("=== Reciprocal Rank Fusion using Node.js + Ollama ===\n");

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

  const allRetrievalResults = [];

  for (let i = 0; i < queryVariations.length; i++) {
    const query = queryVariations[i];

    console.log(`\n=== RESULTS FOR QUERY ${i + 1}: ${query} ===`);

    const results = await vectorStore.similaritySearch(query, 5);
    allRetrievalResults.push(results);

    results.forEach((doc, index) => {
      console.log(`Document ${index + 1}:`);
      console.log(doc.pageContent.slice(0, 180));
      console.log("...\n");
    });

    console.log("-".repeat(50));
  }

  const fusedResults = reciprocalRankFusion(allRetrievalResults, 60);

  console.log("\n" + "=".repeat(60));
  console.log("FINAL RRF RANKING");
  console.log("=".repeat(60));

  fusedResults.slice(0, 10).forEach((item, index) => {
    console.log(`\nRANK ${index + 1} | RRF Score: ${item.score.toFixed(4)}`);
    console.log(item.doc.pageContent.slice(0, 250));
    console.log("-".repeat(50));
  });

  console.log(
    `\n✅ RRF Complete! Fused ${fusedResults.length} unique documents from ${queryVariations.length} query variations.`
  );
}

main().catch((error) => {
  console.error("❌ Error:", error.message);
});