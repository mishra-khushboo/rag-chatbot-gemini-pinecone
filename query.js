import * as dotenv from 'dotenv';
dotenv.config();

import readlineSync from 'readline-sync';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';


async function chatting(question) {
	//convert this question into vector
	const embeddings = new GoogleGenerativeAIEmbeddings({
		apiKey: process.env.GEMINI_API_KEY,
		model: 'text-embedding-004',
	});

	//query vector
	const queryVector = await embeddings.embedQuery(question);

	//make connection with pinecone
	const pinecone = new Pinecone();
	const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

	const searchResults = await pineconeIndex.query({
		topK: 10,
		vector: queryVector,
		includeMetadata: true,
	});

	console.log(searchResults)
}

async function main() {
	const userProblem = readlineSync.question("Ask me anything--> ");
	await chatting(userProblem);
	main();
}


main();