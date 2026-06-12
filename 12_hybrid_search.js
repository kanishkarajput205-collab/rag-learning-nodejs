import { Document } from "@langchain/core/documents";
import { OllamaEmbeddings } from "@langchain/ollama";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";

const chunks = [
  "Microsoft acquired GitHub for 7.5 billion dollars in 2018.",
  "Tesla Cybertruck production ramp begins in 2024.",
  "Google is a large technology company with global operations.",
  "Tesla reported strong quarterly results. Tesla continues to lead in electric vehicles.",
  "SpaceX develops Starship rockets for Mars missions.",
  "The tech giant acquired the code repository platform for software development.",
  "NVIDIA designs Starship architecture for their new GPUs.",
  "Tesla Tesla Tesla financial quarterly results improved significantly.",
  "Cybertruck reservations exceeded company expectations.",
  "Microsoft is a large technology company with global operations.",
  "Apple announced new iPhone features for developers.",
  "The apple orchard harvest was excellent this year.",
  "Python programming language is widely used in AI.",
  "The python snake can grow up to 20 feet long.",
  "Java coffee beans are imported from Indonesia.",
  "Java programming requires understanding of object-oriented concepts.",
  "Orange juice sales increased during winter months.",
  "Orange County reported new housing developments.",
];

function keywordSearch(query, documents, k = 3) {
  const words = query.toLowerCase().split(/\s+/);

  return documents
    .map((doc) => {
      const text = doc.pageContent.toLowerCase();
      let score = 0;

      words.forEach((word) => {
        if (text.includes(word)) score++;
      });

      return { doc, score };
    })
    .filter((item) => item.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, k)
    .map((item) => item.doc);
}

async function main() {
  console.log("=== Hybrid Search using Node.js + Ollama ===\n");

  const documents = chunks.map(
    (chunk, index) =>
      new Document({
        pageContent: chunk,
        metadata: { source: `chunk_${index + 1}` },
      })
  );

  const embeddings = new OllamaEmbeddings({
    model: "nomic-embed-text",
    baseUrl: "http://localhost:11434",
  });

  const vectorStore = await MemoryVectorStore.fromDocuments(
    documents,
    embeddings
  );

  const queries = [
    "space exploration company",
    "Cybertruck",
    "purchase cost 7.5 billion",
    "electric vehicle manufacturing Cybertruck",
    "company performance Tesla",
  ];

  for (const query of queries) {
    console.log("\n" + "=".repeat(60));
    console.log(`Query: ${query}`);
    console.log("=".repeat(60));

    const vectorResults = await vectorStore.similaritySearch(query, 3);
    const keywordResults = keywordSearch(query, documents, 3);

    const combinedMap = new Map();

    vectorResults.forEach((doc, index) => {
      combinedMap.set(doc.pageContent, {
        doc,
        score: 0.7 * (3 - index),
      });
    });

    keywordResults.forEach((doc, index) => {
      const existing = combinedMap.get(doc.pageContent);
      const keywordScore = 0.3 * (3 - index);

      if (existing) {
        existing.score += keywordScore;
      } else {
        combinedMap.set(doc.pageContent, {
          doc,
          score: keywordScore,
        });
      }
    });

    const hybridResults = Array.from(combinedMap.values())
      .sort((a, b) => b.score - a.score)
      .slice(0, 3);

    console.log("\nHybrid Results:");
    hybridResults.forEach((item, index) => {
      console.log(`${index + 1}. ${item.doc.pageContent}`);
    });
  }

  console.log("\n✅ Hybrid Search completed successfully.");
}

main().catch((error) => {
  console.error("❌ Error:", error.message);
});