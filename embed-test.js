import * as dotenv from "dotenv";
dotenv.config();

import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";

const testEmbedding = new GoogleGenerativeAIEmbeddings({
	apiKey: process.env.GEMINI_API_KEY,
	model: "gemini-embedding-001",
});

const vector = await testEmbedding.embedQuery("Hello World");

console.log("Embedding length:", vector.length);