"use client"

import type React from "react"
import { useState, useRef, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { Brain, Search, MessageSquare, Lightbulb, User, X, Moon, Sun, Send } from "lucide-react"
import { useTheme } from "next-themes"
import ReactMarkdown from 'react-markdown'
import { Accordion, AccordionItem, AccordionTrigger, AccordionContent } from '@/components/ui/accordion';
import parse, { domToReact, HTMLReactParserOptions, Element } from 'html-react-parser';

interface Citation {
  id: string
  title: string
  content: string
  relevance?: number
  timestamp?: string
  source?: string
}

interface ResearchMessage {
  id: string
  role: "user" | "assistant"
  content: string
  rationaleHtml?: string
  answerHtml?: string
  thinking?: string
  citations?: Citation[]
  isThinking?: boolean
  isComplete?: boolean
  rationaleStreaming?: boolean
  answerStreaming?: boolean
  clarification?: string
  supervisorPlan?: string
  subagentTasks?: string[]
}

export default function MemoryResearchChatbot() {
  const [messages, setMessages] = useState<ResearchMessage[]>([])
  const [prompt, setPrompt] = useState("")
  const [userId, setUserId] = useState("")
  const [isProcessing, setIsProcessing] = useState(false)
  const [selectedCitation, setSelectedCitation] = useState<Citation | null>(null)
  const [isModalOpen, setIsModalOpen] = useState(false)
  const scrollAreaRef = useRef<HTMLDivElement>(null)
  const { theme, setTheme } = useTheme()
  const [idCounter, setIdCounter] = useState(0)
  const wsRef = useRef<WebSocket | null>(null);
  const [isThinking, setIsThinking] = useState(false);
  const [mounted, setMounted] = useState(false);

  // Prevent hydration mismatch by only rendering theme-dependent content after mount
  useEffect(() => {
    setMounted(true);
  }, []);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollContainer = scrollAreaRef.current.querySelector("[data-radix-scroll-area-viewport]")
      if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight
      }
    }
  }, [messages])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!prompt.trim() || !userId.trim() || isProcessing) return

    const newId = `msg-${idCounter}`
    setIdCounter((c) => c + 1)

    const userMessage: ResearchMessage = {
      id: newId,
      role: "user",
      content: prompt,
      isComplete: true,
    }

    setMessages((prev) => [...prev, userMessage])
    setIsProcessing(true)

    // Create assistant message placeholder
    const assistantMessage: ResearchMessage = {
      id: (Date.now() + 1).toString(),
      role: "assistant",
      content: "",
      thinking: "",
      citations: [],
      isThinking: true,
      isComplete: false,
    }

    setMessages((prev) => [...prev, assistantMessage])
    const currentPrompt = prompt
    setPrompt("")

    try {
      await performSearch(assistantMessage.id, currentPrompt, userId)
    } catch (error) {
      console.error("Search failed:", error)
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessage.id
            ? {
                ...msg,
                content: "Sorry, there was an error processing your request.",
                isComplete: true,
                isThinking: false,
              }
            : msg,
        ),
      )
    } finally {
      setIsProcessing(false)
    }
  }

  const performSearch = async (messageId: string, query: string, userId: string) => {
    setIsThinking(true);
    let rationale = '';
    let answer = '';
    let rationaleStreaming = true;
    let answerStreaming = false;

    // Open or reuse WebSocket
    let ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      ws = new window.WebSocket('ws://localhost:8000/api/v1/multiagent/ws/multiagent');
      wsRef.current = ws;
    }

    ws.onopen = () => {
      ws.send(JSON.stringify({ user_id: userId, prompt: query }));
    };

    ws.onmessage = (event) => {
      let payload: any;
      try {
        payload = JSON.parse(event.data);
      } catch {
        return;
      }
      console.log('WebSocket payload:', payload);
      setMessages((prev) => {
        let found = false;
        const newMessages = prev.map((m) => {
          if (m.id !== messageId) return m;
          found = true;
          let updated = { ...m };
          switch (payload.type) {
            case 'thinking':
              setIsThinking(true);
              updated.isThinking = true;
              break;
            case 'rationale_token':
              rationale += payload.token;
              updated.rationaleHtml = rationale;
              updated.rationaleStreaming = true;
              break;
            case 'rationale_complete':
              updated.rationaleHtml = payload.rationale;
              updated.rationaleStreaming = false;
              break;
            case 'rationale_annotated_html':
              updated.rationaleHtml = payload.rationale_html;
              updated.rationaleStreaming = false;
              break;
            case 'answer_token':
              answer += payload.token;
              updated.answerHtml = answer;
              updated.answerStreaming = true;
              break;
            case 'answer_complete':
              updated.answerHtml = payload.answer;
              updated.answerStreaming = false;
              break;
            case 'answer_annotated_html':
              updated.answerHtml = payload.answer_html;
              updated.answerStreaming = false;
              break;
            case 'citations':
              updated.citations = payload.citations;
              break;
            case 'done':
              updated.isThinking = false;
              setIsThinking(false);
              break;
            case 'clarification': {
              const clarificationText = Array.isArray(payload.clarifications)
                ? payload.clarifications.join('\n\n')
                : String(payload.clarifications);
              updated.clarification = clarificationText;
              updated.content = clarificationText;
              updated.isThinking = false;
              updated.isComplete = true;
              break;
            }
            case 'rationale':
              updated.rationaleHtml = payload.rationale;
              updated.rationaleStreaming = false;
              break;
            case 'answer':
              updated.answerHtml = payload.answer;
              updated.answerStreaming = false;
              break;
            case 'supervisor_plan':
              updated.supervisorPlan = payload.supervisor_plan;
              break;
            case 'subagent_tasks':
              updated.subagentTasks = payload.subagent_tasks;
              break;
          }
          return updated;
        });
        // If not found, add a new assistant message
        if (!found && payload.type !== 'thinking') {
          let newMsg: ResearchMessage = {
            id: messageId,
            role: 'assistant',
            content: '',
            isThinking: false,
            isComplete: true,
            clarification: '',
            rationaleHtml: '',
            answerHtml: '',
            citations: [],
            rationaleStreaming: false,
            answerStreaming: false,
          };
          if (payload.type === 'clarification') {
            const clarificationText = Array.isArray(payload.clarifications)
              ? payload.clarifications.join('\n\n')
              : String(payload.clarifications);
            newMsg.clarification = clarificationText;
            newMsg.content = clarificationText;
          }
          // Add other payload types as needed
          newMessages.push(newMsg);
        }
        return newMessages;
      });
    };

    ws.onerror = (err) => {
      setIsThinking(false);
      setMessages((prev) => prev.map((m) => m.id === messageId ? { ...m, content: 'Sorry, an error occurred.', isThinking: false } : m));
    };

    ws.onclose = () => {
      setIsThinking(false);
    };
  }

  const safeToLower = (val: any) => (typeof val === 'string' ? val.toLowerCase() : '');

  const emptyCitation: Citation = {
    id: '',
    title: '',
    content: '',
    timestamp: '',
    source: '',
  };

  const parseCitationsInText = (text: string, citations: Citation[]) => {
    if (!citations || citations.length === 0) return [{ type: "text", content: text }];

    // Create a map of citation titles to citation objects for quick lookup
    const citationMap = new Map(
      citations.map((c) => [safeToLower(c.title), c])
    );

    // Split text into parts and identify citations
    const parts = [];
    let currentIndex = 0;

    // Look for citation patterns like [Title] or "Title" or Title (case insensitive)
    const citationRegex = /\[([^\]]+)\]|"([^"]+)"|(\b[A-Z][a-zA-Z\s]{2,30}\b)/g;
    let match;

    while ((match = citationRegex.exec(text)) !== null) {
      const citationText = match[1] || match[2] || match[3];
      const citation = citationMap.get(safeToLower(citationText));

      if (citation) {
        // Add text before citation
        if (match.index > currentIndex) {
          parts.push({
            type: "text",
            content: text.slice(currentIndex, match.index),
          });
        }

        // Add citation (ensure all fields are strings)
        parts.push({
          type: "citation",
          content: match[0],
          citation: {
            ...emptyCitation,
            ...citation,
            id: citation.id ? String(citation.id) : '',
            title: citation.title ? String(citation.title) : '',
            content: citation.content ? String(citation.content) : '',
            timestamp: citation.timestamp ? String(citation.timestamp) : '',
            source: citation.source ? String(citation.source) : '',
            relevance: citation.relevance,
          },
        });

        currentIndex = match.index + match[0].length;
      }
    }

    // Add remaining text
    if (currentIndex < text.length) {
      parts.push({
        type: "text",
        content: text.slice(currentIndex),
      });
    }

    return parts.length > 0 ? parts : [{ type: "text", content: text }];
  };

  const handleCitationClick = (citation: Citation) => {
    setSelectedCitation(citation)
    setIsModalOpen(true)
  }

  const clearChat = () => {
    setMessages([])
  }

  function MessageItem({ message, handleCitationClick }: { message: ResearchMessage, handleCitationClick: (citation: Citation) => void }) {
    const rationaleRef = useRef<HTMLDivElement>(null);
    const answerRef = useRef<HTMLDivElement>(null);

    function renderHtmlWithCitations(html: string, citations: Citation[]) {
      const options: HTMLReactParserOptions = {
        replace(domNode) {
          if (
            domNode.type === 'tag' &&
            domNode.name === 'cite' &&
            domNode.attribs &&
            domNode.attribs['data-citation']
          ) {
            const citationIdx = parseInt(domNode.attribs['data-citation'], 10) - 1;
            const citation = citations && citations[citationIdx];
            if (citation) {
              return (
                <span
                  className="underline decoration-purple-400 cursor-pointer bg-purple-100/40 dark:bg-purple-900/40 transition-colors px-1 rounded"
                  onClick={() => handleCitationClick(citation)}
                  title={citation.title}
                >
                  {domToReact(domNode.children as any, options)}
                </span>
              );
            }
          }
        }
      };
      return parse(html, options);
    }

    return (
      <div className="space-y-4">
        {message.role === "user" ? (
          <Card className="ml-auto max-w-2xl">
            <CardContent className="p-4">
              <div className="flex items-start gap-2">
                <MessageSquare className="h-5 w-5 mt-1 text-blue-600 dark:text-blue-400" />
                <p>{message.content}</p>
              </div>
            </CardContent>
          </Card>
        ) : (
          <div className="space-y-4">
            {/* Thinking (Supervisor Plan/Subtasks) Section */}
            {message.supervisorPlan && (
              <Accordion type="single" collapsible defaultValue="supervisor-plan">
                <AccordionItem value="supervisor-plan">
                  <AccordionTrigger>
                    <CardHeader>
                      <CardTitle>Thinking (Supervisor Plan)</CardTitle>
                    </CardHeader>
                  </AccordionTrigger>
                  <AccordionContent>
                    <CardContent>
                      <div className="prose prose-invert max-w-none">
                        <ReactMarkdown>{message.supervisorPlan}</ReactMarkdown>
                      </div>
                    </CardContent>
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
            )}
            {/* Subtasks Section */}
            {message.subagentTasks && message.subagentTasks.length > 0 && (
              <Accordion type="single" collapsible defaultValue="subtasks">
                <AccordionItem value="subtasks">
                  <AccordionTrigger>
                    <CardHeader>
                      <CardTitle>Subtasks</CardTitle>
                    </CardHeader>
                  </AccordionTrigger>
                  <AccordionContent>
                    <CardContent>
                      <ul className="list-disc pl-6">
                        {message.subagentTasks.map((task, idx) => (
                          <li key={idx}>{task}</li>
                        ))}
                      </ul>
                    </CardContent>
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
            )}
            {/* Clarifications Section */}
            {message.clarification && message.clarification.trim() && (
              <Accordion type="single" collapsible defaultValue="clarification">
                <AccordionItem value="clarification">
                  <AccordionTrigger>
                    <CardHeader>
                      <CardTitle>Clarifications</CardTitle>
                    </CardHeader>
                  </AccordionTrigger>
                  <AccordionContent>
                    <CardContent>
                      <div className="prose prose-invert max-w-none">
                        <ReactMarkdown>{message.clarification}</ReactMarkdown>
                      </div>
                    </CardContent>
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
            )}
            {/* Reasoning (Rationale) Section */}
            {message.rationaleHtml && message.rationaleHtml.trim() && (
              <Accordion type="single" collapsible defaultValue="reasoning">
                <AccordionItem value="reasoning">
                  <AccordionTrigger>
                    <CardHeader>
                      <CardTitle>Reasoning</CardTitle>
                    </CardHeader>
                  </AccordionTrigger>
                  <AccordionContent>
                    <CardContent>
                      <div ref={rationaleRef} className="prose prose-invert max-w-none">
                        {message.rationaleStreaming
                          ? message.rationaleHtml
                          : renderHtmlWithCitations(message.rationaleHtml, message.citations || [])}
                      </div>
                    </CardContent>
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
            )}
            {/* Final Answer with Citations Section */}
            {message.answerHtml && message.answerHtml.trim() && (
              <Card>
                <CardHeader>
                  <CardTitle>Final Answer (with Citations)</CardTitle>
                </CardHeader>
                <CardContent>
                  <div ref={answerRef} className="prose prose-invert max-w-none">
                    {message.answerStreaming
                      ? message.answerHtml
                      : renderHtmlWithCitations(message.answerHtml, message.citations || [])}
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen max-w-4xl mx-auto p-4 bg-background text-foreground">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2">
          <Brain className="h-8 w-8 text-blue-600 dark:text-blue-400" />
          <h1 className="text-2xl font-bold">Memory Research Agent</h1>
        </div>

        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={clearChat} disabled={messages.length === 0}>
            Clear Chat
          </Button>
          <div suppressHydrationWarning>
            <Button variant="outline" size="icon" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}> 
              {mounted ? (theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />) : <Moon className="h-4 w-4" />}
            </Button>
          </div>
        </div>
      </div>

      {/* User ID Input */}
      <div className="mb-4">
        <div className="flex items-center gap-2 mb-2">
          <User className="h-4 w-4" />
          <label htmlFor="userId" className="text-sm font-medium">
            User ID
          </label>
        </div>
        <Input
          id="userId"
          value={userId}
          onChange={(e) => setUserId(e.target.value)}
          placeholder="Enter your user ID"
          className="max-w-xs"
        />
      </div>

      {/* Chat Messages */}
      <ScrollArea className="flex-1 mb-4" ref={scrollAreaRef}>
        <div className="space-y-6 pr-4">
          {messages.length === 0 && (
            <div className="text-center text-muted-foreground py-12">
              <Brain className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p className="text-lg mb-2">Welcome to Memory Research Agent</p>
              <p className="text-sm">
                Ask me anything about your memories and I'll search through your personal knowledge base.
              </p>
            </div>
          )}

          {messages.map((message) => (
            <MessageItem key={message.id} message={message} handleCitationClick={handleCitationClick} />
          ))}
        </div>
      </ScrollArea>

      {/* Citation Modal */}
      {isModalOpen && selectedCitation && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
          <div className="bg-background border rounded-lg max-w-2xl w-full max-h-[80vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex items-start justify-between mb-4">
                <h2 className="text-xl font-bold">Memory Details</h2>
                <button
                  onClick={() => setIsModalOpen(false)}
                  className="text-muted-foreground hover:text-foreground transition-colors"
                >
                  <X className="w-6 h-6" />
                </button>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-1">Title</label>
                  <p className="font-medium">{selectedCitation.title}</p>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-1">Memory ID</label>
                  <p className="text-muted-foreground font-mono text-sm">{selectedCitation.id}</p>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-1">Content</label>
                  <div className="bg-muted rounded-lg p-4 border">
                    <p className="whitespace-pre-wrap">{selectedCitation.content}</p>
                  </div>
                </div>

                {selectedCitation.timestamp && (
                  <div>
                    <label className="block text-sm font-medium mb-1">Timestamp</label>
                    <p className="text-muted-foreground">{new Date(selectedCitation.timestamp).toLocaleString()}</p>
                  </div>
                )}

                {selectedCitation.relevance && (
                  <div>
                    <label className="block text-sm font-medium mb-1">Relevance</label>
                    <div className="w-full bg-purple-100 dark:bg-purple-900 rounded-full h-2 mb-2">
                      <div
                        className="bg-purple-500 h-2 rounded-full"
                        style={{ width: `${selectedCitation.relevance * 100}%` }}
                      />
                    </div>
                    <span className="text-sm font-medium">{Math.round(selectedCitation.relevance * 100)}%</span>
                  </div>
                )}

                {selectedCitation.source && (
                  <div>
                    <label className="block text-sm font-medium mb-1">Source</label>
                    <p className="text-muted-foreground">{selectedCitation.source}</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      <form onSubmit={handleSubmit} className="flex items-center gap-2 mt-2">
        <Input
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Ask me anything about your memories..."
          className="flex-1"
          disabled={isProcessing}
        />
        <Button type="submit" disabled={isProcessing || !prompt.trim() || !userId.trim()}>
          <Send className="h-4 w-4 mr-1" /> Send
        </Button>
      </form>

      {isThinking && <div className="w-full h-1 bg-gradient-to-r from-blue-400 to-blue-600 animate-pulse" />}
    </div>
  )
}
