import * as dotenv from 'dotenv';
dotenv.config();
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';




async function indexDocument() {

	//load pdf  phase 1
	const PDF_PATH = './dsa.pdf';
	const pdfLoader = new PDFLoader(PDF_PATH);
	const rawDocs = await pdfLoader.load();
	console.log("PDF loaded");
	console.log("Raw docs:", rawDocs.length);
	console.log(rawDocs[0]);

	//chunking
	const textSplitter = new RecursiveCharacterTextSplitter({
		chunkSize: 1000,
		chunkOverlap: 200,
	});
	const chunkedDocs = await textSplitter.splitDocuments(rawDocs);
	console.log("Chunking Completed");
	console.log("Number of chunks:", chunkedDocs.length);
	console.log(chunkedDocs[0]);

	console.log("GOOGLE_API_KEY =", process.env.GOOGLE_API_KEY);
	console.log("GEMINI_API_KEY =", process.env.GEMINI_API_KEY);
	console.log("PINECONE_API_KEY =", process.env.PINECONE_API_KEY);
	//vector Embedding model
	const embeddings = new GoogleGenerativeAIEmbeddings({
		apiKey: process.env.GEMINI_API_KEY,
		model: "gemini-embedding-001",
	});

	const vec = await embeddings.embedQuery("test");

	console.log("Vector length:", vec.length);
	console.log("GEMINI:", process.env.GEMINI_API_KEY);
	console.log("PINECONE:", process.env.PINECONE_API_KEY);
	console.log("Embedding model configured");

	//configure database
	//Initialize Pinecone Client
	const testEmbedding = await embeddings.embedQuery("Hello world");

	console.log("Embedding length:", testEmbedding.length);
	console.log(testEmbedding.slice(0, 5));
	const pinecone = new Pinecone({
		apiKey: process.env.PINECONE_API_KEY,
	});

	const pineconeIndex = pinecone.Index(
		process.env.PINECONE_INDEX_NAME
	);
	const testVector = await embeddings.embedQuery("test");

	console.log(
		"Embedding dimension:",
		testVector.length
	);

	const vectorStore = new PineconeStore(
		embeddings,
		{
			pineconeIndex,
		}
	);
	console.log("First chunk:");
	console.log(chunkedDocs[0].pageContent.substring(0, 100));

	await vectorStore.addDocuments(chunkedDocs);

	console.log("Data Stored successfully");

}



indexDocument();