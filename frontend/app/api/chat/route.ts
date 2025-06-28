import { openai } from "@ai-sdk/openai"
import { streamText } from "ai"

export const maxDuration = 30

export async function POST(req: Request) {
  const { messages } = await req.json()

  const result = streamText({
    model: openai("gpt-4o"),
    messages: messages.map((msg: any) => ({
      role: msg.role,
      content: msg.content,
    })),
    system: `You are a memory research agent. You have access to a comprehensive database of memories and research about human memory, cognition, and neuroscience. 

When responding to queries:
1. First, think through the problem step by step
2. Search through relevant memories and research
3. Provide citations to specific memories/research
4. Give a comprehensive, well-researched answer

Your responses should be thorough, scientific, and backed by the memory database.`,
  })

  return result.toDataStreamResponse()
}
