import { ChatOllama } from "@langchain/ollama";

const teslaText = `Tesla's Q3 Results
Tesla reported record revenue of $25.2B in Q3 2024.
The company exceeded analyst expectations by 15%.
Revenue growth was driven by strong vehicle deliveries.

Model Y Performance
The Model Y became the best-selling vehicle globally, with 350,000 units sold.
Customer satisfaction ratings reached an all-time high of 96%.
Model Y now represents 60% of Tesla's total vehicle sales.

Production Challenges
Supply chain issues caused a 12% increase in production costs.
Tesla is working to diversify its supplier base.
New manufacturing techniques are being implemented to reduce costs.`;

async function main() {
  const llm = new ChatOllama({
    model: "llama3",
    baseUrl: "http://localhost:11434",
    temperature: 0,
  });

  const prompt = `
You are a text chunking expert.

Split the following text into logical chunks.

Rules:
- Each chunk should be around 200 characters or less
- Split at natural topic boundaries
- Keep related information together
- Put <<<SPLIT>>> between chunks

Text:
${teslaText}

Return only the text with <<<SPLIT>>> markers.
`;

  console.log("🤖 Asking Ollama to chunk the text...\n");

  const response = await llm.invoke(prompt);

  const markedText = response.content;

  const chunks = markedText
    .split("<<<SPLIT>>>")
    .map((chunk) => chunk.trim())
    .filter(Boolean);

  console.log("🎯 AGENTIC CHUNKING RESULTS:");
  console.log("=".repeat(50));

  chunks.forEach((chunk, index) => {
    console.log(`Chunk ${index + 1}: (${chunk.length} chars)`);
    console.log(`"${chunk}"`);
    console.log();
  });
}

main().catch((error) => {
  console.error("❌ Error:", error.message);
});