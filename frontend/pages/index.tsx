import { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import { Send, Loader2, RefreshCw, LogOut, User, History, ChevronDown, ChevronUp, BookOpen, Activity, ThumbsUp, ThumbsDown, Moon, Sun, Copy, Check, ShoppingBag, ExternalLink, X, Search, ArrowDown } from 'lucide-react';
import Head from 'next/head';
import { useSession, Citation, SourceReference, FlowMetadata, UserProfile, Product, AuthExpiredError } from '../hooks/useSession';
import { useAuth } from '../hooks/useAuth';
import dynamic from 'next/dynamic';
const LoginPage = dynamic(() => import('../components/LoginPage'), { ssr: false });
const TTSButton = dynamic(() => import('../components/TTSButton'), { ssr: false });
import { useTheme } from '../hooks/useTheme';
import { useToast } from '../components/Toast';
import { parseResponseForVerses, ParsedSegment } from '../utils/parseResponseForVerses';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

/* ============================================================================
   Helpers
   ============================================================================ */

/**
 * Fix inline bullets — inserts newlines before "- " that appears mid-line.
 * Handles: (a) old stored conversations generated before the markdown change,
 * (b) edge cases where Gemini emits bullets on one line despite the prompt
 * saying "each bullet on its own line". Without this, react-markdown treats
 * inline "- " as plain text instead of rendering a <ul><li> list.
 */
function fixInlineBullets(text: string): string {
  // Match "- " preceded by sentence-ending punctuation or a word char and
  // optional whitespace — i.e. a bullet marker that isn't already at line
  // start. Replace with a newline before the dash.
  return text.replace(/([.!?:;)\]'"a-zA-Z0-9])\s+- /g, '$1\n- ');
}

/* ============================================================================
   Components
   ============================================================================ */

/**
 * Product Card Component
 */
// Fix 25: Validate product URLs to prevent javascript: or data: XSS
function safeProductUrl(url?: string): string {
  if (!url) return 'https://my3ionetra.com';
  try {
    const parsed = new URL(url);
    if (['https:', 'http:'].includes(parsed.protocol)) return url;
  } catch {}
  return 'https://my3ionetra.com';
}

function ProductCard({ product }: { product: Product }) {
  return (
    <a
      href={safeProductUrl(product.product_url)}
      target="_blank"
      rel="noopener noreferrer"
      className="flex flex-col bg-white dark:bg-gray-800 border border-gray-100 dark:border-gray-700 rounded-2xl overflow-hidden shadow-sm hover:shadow-md transition-all group w-[180px] shrink-0 cursor-pointer no-underline"
    >
      <div className="h-28 bg-gray-50 dark:bg-gray-700 relative overflow-hidden">
        {product.image_url ? (
          <img
            src={product.image_url}
            alt={product.name}
            loading="lazy"
            onError={(e) => { e.currentTarget.style.display = 'none'; }}
            onLoad={(e) => { e.currentTarget.style.opacity = '1'; }}
            className="w-full h-full object-cover group-hover:scale-110 transition-all duration-500"
            style={{ opacity: 0, transition: 'opacity 0.3s ease' }}
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-gray-300 dark:text-gray-600">
            <ShoppingBag className="w-8 h-8" />
          </div>
        )}
      </div>
      <div className="p-3 flex flex-col flex-1">
        <h4 className="text-[11px] font-black text-gray-900 dark:text-gray-100 leading-tight mb-1 truncate" title={product.name}>{product.name}</h4>
        <p className="text-[9px] font-bold text-orange-600 uppercase tracking-tighter mb-2">{product.category}</p>

        <div className="mt-auto flex items-center justify-between gap-2">
          <span className="text-[11px] font-black text-gray-900 dark:text-gray-100">{product.currency === 'INR' ? '₹' : product.currency}{Number(product.amount).toLocaleString('en-IN')}</span>
          <span className="p-1.5 bg-gray-900 dark:bg-gray-600 text-white rounded-lg group-hover:bg-orange-600 transition-colors">
            <ExternalLink className="w-3 h-3" />
          </span>
        </div>
      </div>
    </a>
  );
}

/**
 * Product Recommendations Horizontal Section
 */
function ProductDisplay({ products }: { products: Product[] }) {
  if (!products || products.length === 0) return null;

  return (
    <div className="mt-4 pt-4 border-t border-orange-50/50 dark:border-gray-700">
      <div className="flex items-center gap-2 mb-3">
        <div className="p-1.5 bg-orange-100/50 dark:bg-orange-900/30 rounded-lg">
          <ShoppingBag className="w-3 h-3 text-orange-600" />
        </div>
        <span className="text-[9px] font-black text-orange-900 dark:text-orange-300 uppercase tracking-widest">
          Recommended for your journey
        </span>
      </div>
      <div className="relative">
        <div className="flex gap-3 overflow-x-auto pb-2 scrollbar-hide -mx-1 px-1 snap-x">
          {products.map((product) => (
            <ProductCard key={product.product_url || product.name} product={product} />
          ))}
          <a
            href="https://my3ionetra.com"
            target="_blank"
            rel="noopener noreferrer"
            className="flex flex-col items-center justify-center bg-orange-50/30 dark:bg-gray-800/50 border border-dashed border-orange-200 dark:border-orange-800 rounded-2xl p-4 w-[120px] shrink-0 hover:bg-orange-50 dark:hover:bg-gray-700 transition-all group"
          >
            <div className="w-8 h-8 bg-white dark:bg-gray-700 rounded-full flex items-center justify-center mb-2 shadow-sm border border-orange-100 dark:border-gray-600 group-hover:scale-110 transition-transform">
              <ExternalLink className="w-4 h-4 text-orange-600" />
            </div>
            <span className="text-[9px] font-black text-orange-800 dark:text-orange-300 uppercase text-center leading-tight">Visit<br />Netra Store</span>
          </a>
        </div>
        <div className="absolute right-0 top-0 bottom-2 w-8 bg-gradient-to-l from-white dark:from-gray-900 pointer-events-none" />
      </div>
    </div>
  );
}

const rawApiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';
const API_URL = rawApiUrl.endsWith('/') ? rawApiUrl.slice(0, -1) : rawApiUrl;

interface Message {
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  sources?: SourceReference[];
  flowMetadata?: FlowMetadata;
  recommendedProducts?: Product[];
  timestamp: Date;
  isWelcome?: boolean;
  isError?: boolean;
}

interface ConversationSummary {
  id: string;
  session_id?: string;
  title: string;
  created_at: string;
  message_count: number;
}


export default function Home() {
  const { theme, toggleTheme } = useTheme();
  const { user, isAuthenticated, login, register, logout, isLoading: authLoading, error: authError, getAuthHeader } = useAuth();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const language = 'en';
  const [isProcessing, setIsProcessing] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState<string>('');
  const [useConversationFlow, setUseConversationFlow] = useState(true);
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);
  const [history, setHistory] = useState<ConversationSummary[]>([]);
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const [historyLimit, setHistoryLimit] = useState(30);  // Fix 16: paginated history
  const [historyLoading, setHistoryLoading] = useState(false);  // Fix 18: loading state
  const [historyHasMore, setHistoryHasMore] = useState(false);
  const [historyOffset, setHistoryOffset] = useState(0);
  const HISTORY_LIMIT = 20;
  const [feedback, setFeedback] = useState<Record<number, 'like' | 'dislike'>>({});
  const [expandedCitations, setExpandedCitations] = useState<Record<number, boolean>>({});
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [historySearch, setHistorySearch] = useState('');
  const [showScrollBottom, setShowScrollBottom] = useState(false);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const sidebarRef = useRef<HTMLElement>(null);
  const { toast } = useToast();

  const handleRetry = (errorIndex: number) => {
    let found = false;
    for (let i = errorIndex - 1; i >= 0; i--) {
      if (messages[i].role === 'user') {
        setInput(messages[i].content);
        found = true;
        break;
      }
    }
    if (found) {
      setMessages(prev => prev.filter((_, idx) => idx !== errorIndex));
    }
  };

  // Copy with visual confirmation and toast
  const handleCopy = useCallback((text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    toast('Copied to clipboard');
    setTimeout(() => setCopiedId(null), 2000);
  }, [toast]);

  // Scroll-to-bottom detection
  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container) return;
    const onScroll = () => {
      const distFromBottom = container.scrollHeight - container.scrollTop - container.clientHeight;
      setShowScrollBottom(distFromBottom > 300);
    };
    container.addEventListener('scroll', onScroll, { passive: true });
    return () => container.removeEventListener('scroll', onScroll);
  }, []);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  // Time-of-day greeting
  const greeting = useMemo(() => {
    const hour = new Date().getHours();
    const name = user?.name?.split(' ')[0] || '';
    const prefix = hour < 12 ? 'Good Morning' : hour < 17 ? 'Good Afternoon' : 'Good Evening';
    return name ? `${prefix}, ${name}` : prefix;
  }, [user?.name]);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isHistoryOpen) {
        setIsHistoryOpen(false);
      }
      if ((e.metaKey || e.ctrlKey) && e.key === '/') {
        e.preventDefault();
        textareaRef.current?.focus();
      }
      if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key === 'n') {
        e.preventDefault();
        handleNewConversation();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [isHistoryOpen]);

  // Focus trap for mobile sidebar
  useEffect(() => {
    if (!isHistoryOpen) return;
    const sidebar = sidebarRef.current;
    if (!sidebar) return;
    const focusable = sidebar.querySelectorAll<HTMLElement>('button, input, [tabindex]:not([tabindex="-1"])');
    if (focusable.length === 0) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    const trap = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return;
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    };
    // Only trap on mobile (check if sidebar is fixed/overlay)
    const isMobile = window.innerWidth < 1024;
    if (isMobile) {
      sidebar.addEventListener('keydown', trap);
      first.focus();
      return () => sidebar.removeEventListener('keydown', trap);
    }
  }, [isHistoryOpen]);

  useEffect(() => {
    if (authError) toast(authError, 'error');
  }, [authError]);

  // Fix 14: Only update feedback UI after successful API call
  const handleFeedback = async (messageIndex: number, responseText: string, type: 'like' | 'dislike') => {
    const prev = feedback[messageIndex];
    if (prev === type) return;
    try {
      const headers: Record<string, string> = { 'Content-Type': 'application/json' };
      const auth = getAuthHeader();
      if (auth) Object.assign(headers, auth);
      const res = await fetch(`${API_URL}/api/feedback`, {
        method: 'POST',
        headers,
        credentials: 'include',
        body: JSON.stringify({
          session_id: session.sessionId || '',
          message_index: messageIndex,
          response_text: responseText,
          feedback: type,
        }),
      });
      if (res.ok) {
        // Only update UI after successful API save
        setFeedback((f) => ({ ...f, [messageIndex]: type }));
        toast(type === 'like' ? 'Thanks for your feedback!' : 'Feedback noted, we\'ll improve', 'info');
      } else {
        toast('Could not save feedback', 'error');
      }
    } catch (err) {
      if (process.env.NODE_ENV === 'development') console.error('Feedback submit failed:', err);
      toast('Could not save feedback', 'error');
    }
  };

  const userProfile: UserProfile | undefined = user ? {
    age_group: user.age_group || '',
    gender: user.gender || '',
    profession: user.profession || '',
    name: user.name || '',
    preferred_deity: user.preferred_deity || '',
    rashi: user.rashi || '',
    gotra: user.gotra || '',
    nakshatra: user.nakshatra || '',
  } : undefined;

  const {
    session,
    isLoading: sessionLoading,
    sendMessage,
    sendMessageStream,
    resetSession,
    loadSession,
    error: sessionError,
  } = useSession(userProfile, getAuthHeader());

  const parsedMessages = useMemo(
    () => messages.map(m => m.role === 'assistant' && m.content ? parseResponseForVerses(m.content) : null),
    [messages]
  );

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesRef = useRef<Message[]>(messages);
  messagesRef.current = messages;
  const lastScrollRef = useRef<number>(0);
  const isStreamingRef = useRef<boolean>(false);

  useEffect(() => {
    const now = Date.now();
    // During streaming, throttle scroll to every 200ms to avoid jank
    // After streaming, scroll immediately
    const throttle = isStreamingRef.current ? 200 : 0;
    if (now - lastScrollRef.current < throttle) return;
    lastScrollRef.current = now;
    messagesEndRef.current?.scrollIntoView({ behavior: isStreamingRef.current ? 'auto' : 'smooth' });
  }, [messages, isProcessing]);


  // Auto-save conversation after streaming completes (not during)
  // Fix 13: Use refs for saveConversation/fetchHistory to avoid stale closures
  const saveRef = useRef<((id?: string) => Promise<void>) | null>(null);
  const fetchHistoryRef = useRef<(() => Promise<void>) | null>(null);

  useEffect(() => {
    if (!isAuthenticated || messages.length < 2 || isProcessing) return;
    const convId = currentConversationId || session.sessionId;
    if (!convId) return;

    const timer = setTimeout(async () => {
      try {
        await saveRef.current?.(convId);
        await fetchHistoryRef.current?.();
      } catch (err) {
        if (process.env.NODE_ENV === 'development') console.error('Auto-save failed:', err);
      }
    }, 2000);

    return () => clearTimeout(timer);
  }, [messages, currentConversationId, session.sessionId, isAuthenticated, isProcessing]);

  useEffect(() => {
    setMessages([]);
    setCurrentConversationId(null);
    if (isAuthenticated) {
      fetchHistory();
    }
  }, [user?.id, isAuthenticated]);

  // Re-validate auth when tab becomes visible after being backgrounded
  // Fix 12: Use ref-based handler to avoid re-registering on every getAuthHeader change
  const visibilityHandlerRef = useRef<() => void>();
  visibilityHandlerRef.current = async () => {
    if (document.visibilityState !== 'visible') return;
    const authHdr = getAuthHeader();
    if (!authHdr) return;
    try {
      const res = await fetch(`${API_URL}/api/auth/verify`, {
        headers: authHdr,
        credentials: 'include',
      });
      if (res.status === 401) logout();
    } catch {
      // Network error — don't logout, could be transient
    }
  };
  useEffect(() => {
    if (!isAuthenticated) return;
    const handler = () => visibilityHandlerRef.current?.();
    document.addEventListener('visibilitychange', handler);
    return () => document.removeEventListener('visibilitychange', handler);
  }, [isAuthenticated]);

  // Fix 18: Add loading state for history fetch
  const fetchHistory = async () => {
    if (!isAuthenticated) return;
    setHistoryLoading(true);
    try {
      const res = await fetch(`${API_URL}/api/user/conversations?limit=${HISTORY_LIMIT}&offset=0`, {
        headers: getAuthHeader(),
        credentials: 'include',
      });
      if (res.ok) {
        const data = await res.json();
        const conversations = Array.isArray(data.conversations) ? data.conversations : [];
        setHistory(conversations);
        setHistoryHasMore(conversations.length === HISTORY_LIMIT);
        setHistoryOffset(HISTORY_LIMIT);
      } else if (res.status === 401) {
        localStorage.removeItem('auth_user');
        logout();
      }
    } catch (error) {
      if (process.env.NODE_ENV === 'development') console.error('Failed to fetch history:', error);
    } finally {
      setHistoryLoading(false);
    }
  };

  const loadMoreHistory = async () => {
    if (!isAuthenticated || historyLoading || !historyHasMore) return;
    setHistoryLoading(true);
    try {
      const res = await fetch(`${API_URL}/api/user/conversations?limit=${HISTORY_LIMIT}&offset=${historyOffset}`, {
        headers: getAuthHeader(),
        credentials: 'include',
      });
      if (res.ok) {
        const data = await res.json();
        const more = Array.isArray(data.conversations) ? data.conversations : [];
        setHistory(prev => [...prev, ...more]);
        setHistoryHasMore(more.length === HISTORY_LIMIT);
        setHistoryOffset(prev => prev + more.length);
      }
    } catch (error) {
      if (process.env.NODE_ENV === 'development') console.error('Failed to load more history:', error);
    } finally {
      setHistoryLoading(false);
    }
  };

  const handleSelectSession = async (sessionId: string) => {
    try {
      setIsProcessing(true);
      const data = await loadSession(sessionId);
      if (data && data.messages) {
        setMessages(data.messages.map((m: any) => ({
          ...m,
          timestamp: new Date(m.timestamp)
        })));
        setCurrentConversationId(data.session_id || sessionId);
        setIsHistoryOpen(false);
        setFeedback({});
        setExpandedCitations({});
      }
    } catch (error) {
      if (process.env.NODE_ENV === 'development') console.error('Failed to switch session:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  const saveConversation = async (overrideId?: string) => {
    if (!isAuthenticated || messages.length <= 1) return;
    const convId = overrideId || currentConversationId || session.sessionId;
    if (!convId) return;
    try {
      const title = messages.find(m => m.role === 'user')?.content.slice(0, 50) || 'Conversation';
      await fetch(`${API_URL}/api/user/conversations`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...getAuthHeader(),
        },
        credentials: 'include',
        body: JSON.stringify({
          conversation_id: convId,
          title,
          messages: messages.map(m => ({
            role: m.role,
            content: m.content,
            citations: m.citations,
            timestamp: m.timestamp.toISOString(),
          })),
        }),
      });
    } catch (error) {
      if (process.env.NODE_ENV === 'development') console.error('Failed to save conversation:', error);
    }
  };

  // Fix 13: Keep refs in sync with latest function references
  saveRef.current = saveConversation;
  fetchHistoryRef.current = fetchHistory;

  const handleNewConversation = async () => {
    if (messages.length > 1) {
      await saveConversation();
    }
    resetSession();
    setMessages([]);
    setCurrentConversationId(null);
    setFeedback({});
    setExpandedCitations({});
    setHistorySearch('');
    fetchHistory();
    toast('New session started', 'info');
  };

  const handleTextSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isProcessing) return;

    const userMessage: Message = {
      role: 'user',
      content: input,
      timestamp: new Date(),
    };

    const currentInput = input;
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    // Reset textarea height after clearing
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
    setIsProcessing(true);

    try {
      if (useConversationFlow) {
        // Add empty placeholder for streaming text
        const placeholder: Message = { role: 'assistant', content: '', timestamp: new Date() };
        setMessages((prev) => [...prev, placeholder]);

        let accumulatedText = '';
        isStreamingRef.current = true;

        await sendMessageStream(
          currentInput,
          language,
          // onToken — update message content directly; React 18 auto-batches
          // synchronous calls within the same reader.read() chunk, and renders
          // between async chunks for progressive streaming display
          (text) => {
            accumulatedText += text;
            const snapshot = accumulatedText;
            setMessages((prev) => {
              const updated = [...prev];
              updated[updated.length - 1] = { ...updated[updated.length - 1], content: snapshot };
              return updated;
            });
          },
          // onMetadata — update session state
          (meta) => {
            if (!currentConversationId && meta.session_id) {
              setCurrentConversationId(meta.session_id);
            }
          },
          // onDone — finalize with clean text, products, citations
          (final) => {
            isStreamingRef.current = false;
            setLoadingStatus('');
            const flowMetadata = final.flow_metadata || undefined;
            setMessages((prev) => {
              const updated = [...prev];
              // Attach flowMetadata to the last user message
              for (let i = updated.length - 1; i >= 0; i--) {
                if (updated[i].role === 'user') {
                  updated[i] = { ...updated[i], flowMetadata };
                  break;
                }
              }
              // Finalize assistant message
              updated[updated.length - 1] = {
                ...updated[updated.length - 1],
                content: final.full_response || accumulatedText,
                recommendedProducts: final.recommended_products || [],
                flowMetadata,
                citations: final.citations || [],
                sources: final.sources || [],
              };
              return updated;
            });
          },
          // onError — fallback to non-streaming sendMessage()
          async (err) => {
            isStreamingRef.current = false;
            setLoadingStatus('');
            if (err instanceof AuthExpiredError) { logout(); return; }
            if (process.env.NODE_ENV === 'development') console.warn('Stream failed, falling back:', err);
            // Remove placeholder
            setMessages((prev) => prev.slice(0, -1));
            try {
              const response = await sendMessage(currentInput, language);
              if (!response) throw new Error('No response received');
              let responseText = typeof response.response === 'string' ? response.response : '';
              if (!responseText.trim()) responseText = 'I apologize, but I received an invalid response. Please try again.';
              const flowMetadata = response.flow_metadata || undefined;
              const assistantMessage: Message = {
                role: 'assistant',
                content: responseText,
                citations: response.citations || [],
                sources: response.sources || [],
                recommendedProducts: response.recommended_products || [],
                flowMetadata,
                timestamp: new Date(),
              };
              setMessages((prev) => {
                const nextMessages = [...prev];
                for (let i = nextMessages.length - 1; i >= 0; i--) {
                  if (nextMessages[i].role === 'user') {
                    nextMessages[i] = { ...nextMessages[i], flowMetadata };
                    break;
                  }
                }
                return [...nextMessages, assistantMessage];
              });
              if (!currentConversationId && response.session_id) {
                setCurrentConversationId(response.session_id);
              }
            } catch (fallbackErr: any) {
              if (process.env.NODE_ENV === 'development') {
                console.error('Fallback also failed:', fallbackErr);
              }
              if (fallbackErr instanceof AuthExpiredError) { logout(); return; }
              // Fix 7: Show meaningful error to user instead of generic message
              const errorDetail = fallbackErr?.message || 'Unknown error';
              const isNetworkError = errorDetail.includes('fetch') || errorDetail.includes('network') || errorDetail.includes('abort');
              setMessages((prev) => [...prev, {
                role: 'assistant',
                content: isNetworkError
                  ? 'It seems the connection was interrupted. Please check your network and try again.'
                  : 'I apologize, I encountered a temporary issue. Please try sending your message again.',
                timestamp: new Date(),
                isError: true,
              }]);
            }
          },
          // onStatus — update loading indicator text
          (status) => { setLoadingStatus(status.message); },
        );
      }
    } catch (error) {
      if (process.env.NODE_ENV === 'development') console.error('Error in handleTextSubmit:', error);
      if (error instanceof AuthExpiredError) { logout(); return; }
      // Remove placeholder if it exists
      setMessages((prev) => {
        if (prev.length > 0 && prev[prev.length - 1].role === 'assistant' && prev[prev.length - 1].content === '') {
          return prev.slice(0, -1);
        }
        return prev;
      });
      const errorMessage: Message = {
        role: 'assistant',
        content: 'I apologize, but I encountered an error. Please try again.',
        timestamp: new Date(),
        isError: true,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsProcessing(false);
      setLoadingStatus('');
    }
  };

  if (!isAuthenticated) {
    return (
      <>
        <Head>
          <title>3ioNetra Spiritual Companion - Login</title>
          <meta name="viewport" content="width=device-width, initial-scale=1" />
        </Head>
        <LoginPage onLogin={login} onRegister={register} isLoading={authLoading} error={authError} />
      </>
    );
  }

  return (
    <>
      <Head>
        <title>3ioNetra Spiritual Companion</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <style>{`
          @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
          .animate-fade-in { animation: fadeIn 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards; }
          .scrollbar-hide::-webkit-scrollbar { display: none; }
          .scrollbar-hide { -ms-overflow-style: none; scrollbar-width: none; }
          @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.2; } }
          .streaming-cursor { display: inline-block; width: 2px; height: 1em; background: #c2410c; margin-left: 2px; vertical-align: text-bottom; animation: blink 1s ease-in-out infinite; }
        `}</style>
      </Head>

      <main className="h-[100dvh] bg-[#fcfcfc] dark:bg-[#0a0a0a] flex overflow-hidden font-['Inter']">
        <aside ref={sidebarRef} data-testid="sidebar" className={`fixed inset-y-0 left-0 z-50 w-72 bg-white/90 dark:bg-gray-900/95 backdrop-blur-xl border-r border-orange-100 dark:border-gray-800 shadow-2xl dark:shadow-black/30 transform transition-all duration-500 ease-in-out lg:relative ${isHistoryOpen ? 'translate-x-0' : '-translate-x-full lg:-ml-72'}`}>
          <div className="h-full flex flex-col">
            <div className="p-5 border-b border-orange-100/50 dark:border-gray-800 flex items-center justify-between">
              <h2 className="font-black text-gray-900 dark:text-gray-100 flex items-center gap-2 text-lg tracking-tight">
                <History className="w-4 h-4 text-orange-600" />
                Conversations
              </h2>
              <button onClick={() => setIsHistoryOpen(false)} aria-label="Close conversation history" className="p-2 text-gray-400 hover:text-orange-600 hover:bg-orange-50 dark:hover:bg-gray-800 rounded-xl transition-all">
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="px-3 pt-3 pb-1">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-gray-400" />
                <input
                  type="text"
                  value={historySearch}
                  onChange={(e) => setHistorySearch(e.target.value)}
                  placeholder="Search conversations..."
                  aria-label="Search conversation history"
                  className="w-full pl-9 pr-3 py-2.5 bg-gray-50/50 dark:bg-gray-800/50 border border-orange-100 dark:border-gray-700 rounded-xl text-xs font-bold text-gray-900 dark:text-gray-100 placeholder:text-gray-400 dark:placeholder:text-gray-500 focus:ring-2 focus:ring-orange-500/10 focus:border-orange-200 dark:focus:border-orange-700 outline-none transition-all"
                />
                {historySearch && (
                  <button onClick={() => setHistorySearch('')} className="absolute right-2 top-1/2 -translate-y-1/2 p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300">
                    <X className="w-3 h-3" />
                  </button>
                )}
              </div>
            </div>
            <div className="flex-1 overflow-y-auto p-3 space-y-1.5 scrollbar-hide">
              {/* Fix 18: Loading state */}
              {historyLoading && (
                <div className="space-y-2 p-3">
                  {[0,1,2,3].map(i => (
                    <div key={i} className="h-14 rounded-xl bg-orange-50/50 dark:bg-gray-800 animate-pulse" />
                  ))}
                </div>
              )}
              {!historyLoading && history.length === 0 && (
                <p className="text-xs text-gray-400 dark:text-gray-500 p-6 text-center italic">No conversations yet. Start a chat to see your history here.</p>
              )}
              {/* Fix 16: Paginated history — show first 30, load more on demand */}
              {(() => {
                const filtered = history.filter(item => !historySearch || item.title.toLowerCase().includes(historySearch.toLowerCase()));
                const visible = filtered.slice(0, historyLimit);
                if (!historyLoading && historySearch && visible.length === 0 && history.length > 0) {
                  return <p className="text-xs text-gray-400 dark:text-gray-500 p-6 text-center italic">No conversations match &ldquo;{historySearch}&rdquo;</p>;
                }
                let lastGroup: string | null = null;
                return visible.map((item) => {
                  const diff = Math.floor((Date.now() - new Date(item.created_at).getTime()) / 86400000);
                  const group = diff === 0 ? 'Today' : diff === 1 ? 'Yesterday' : diff < 7 ? 'This Week' : 'Earlier';
                  const showHeader = group !== lastGroup;
                  lastGroup = group;
                  return (
                    <div key={item.id}>
                      {showHeader && (
                        <p className="text-[9px] font-black uppercase tracking-widest text-gray-300 dark:text-gray-600 px-2 pt-3 pb-1">{group}</p>
                      )}
                      <button
                        onClick={() => handleSelectSession(item.session_id || item.id)}
                        className={`w-full p-4 text-left rounded-2xl transition-all border ${currentConversationId === (item.session_id || item.id) ? 'bg-orange-50 dark:bg-orange-900/20 border-orange-200 dark:border-orange-800 shadow-sm' : 'border-transparent hover:bg-gray-50 dark:hover:bg-gray-800'}`}
                      >
                        <p className="text-sm font-bold text-gray-900 dark:text-gray-100 truncate mb-0.5">{item.title}</p>
                        <div className="flex items-center justify-between opacity-50">
                          <span className="text-[9px] font-bold">{new Date(item.created_at).toLocaleDateString()}</span>
                          <span className="text-[9px] font-bold uppercase">{item.message_count} msgs</span>
                        </div>
                      </button>
                    </div>
                  );
                });
              })()}
              {history.length > historyLimit && (
                <button
                  onClick={() => setHistoryLimit(prev => prev + 30)}
                  className="w-full p-3 text-xs font-bold text-orange-600 dark:text-orange-400 hover:bg-orange-50 dark:hover:bg-gray-800 rounded-xl transition-all"
                >
                  Load more ({historyLimit} of {history.length})
                </button>
              )}
              {historyHasMore && (
                <button
                  onClick={loadMoreHistory}
                  disabled={historyLoading}
                  className="w-full p-3 text-xs font-bold text-orange-600 dark:text-orange-400 hover:bg-orange-50 dark:hover:bg-gray-800 rounded-xl transition-all disabled:opacity-50"
                >
                  {historyLoading ? 'Loading...' : 'Load more'}
                </button>
              )}
            </div>

            <div className="p-5 border-t border-orange-100 dark:border-gray-800 bg-orange-50/10 dark:bg-gray-800/30">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-gradient-to-br from-orange-100 to-amber-100 dark:from-orange-900/50 dark:to-amber-900/50 rounded-xl flex items-center justify-center border border-white dark:border-gray-700 shadow-sm ring-1 ring-orange-200/50 dark:ring-orange-800/50">
                  <User className="w-5 h-5 text-orange-600" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-black text-gray-900 dark:text-gray-100 truncate">{user?.name}</p>
                  <button onClick={logout} className="text-[10px] text-orange-600 dark:text-orange-400 font-black uppercase tracking-tighter hover:underline">Sign Out</button>
                </div>
              </div>
            </div>
          </div>
        </aside>

        {isHistoryOpen && <div className="fixed inset-0 bg-black/10 backdrop-blur-sm z-40 lg:hidden" onClick={() => setIsHistoryOpen(false)} />}

        <div className="flex-1 flex flex-col h-screen overflow-hidden relative">
          <header className="sticky top-0 z-30 bg-white/70 dark:bg-gray-900/70 backdrop-blur-md border-b border-orange-50 dark:border-gray-800 shrink-0 px-4 py-2.5">
            <div className="max-w-4xl mx-auto flex items-center justify-between">
              <div className="flex items-center gap-3">
                <button data-testid="sidebar-toggle" aria-label={isHistoryOpen ? "Close conversation history" : "Open conversation history"} onClick={() => setIsHistoryOpen(!isHistoryOpen)} className="p-2 text-gray-600 dark:text-gray-400 hover:text-orange-600 hover:bg-orange-50 dark:hover:bg-gray-800 rounded-xl transition-all shadow-sm border border-orange-50 dark:border-gray-700 bg-white dark:bg-gray-800">
                  <History className={`w-4 h-4 transition-transform duration-500 ${isHistoryOpen ? 'rotate-180' : ''}`} />
                </button>
                <div className="flex flex-col">
                  <h1 className="text-xl font-black flex items-center gap-2 tracking-tighter">
                    <img src="/logo-circle.jpg" alt="3ioNetra" className="w-8 h-8 rounded-full shadow-sm" />
                    <span className="text-[#1a2b56] dark:text-white">3ioNetra</span>
                    <span className="text-[8px] font-black uppercase tracking-widest px-1.5 py-0.5 bg-orange-100 dark:bg-orange-900/40 text-orange-600 dark:text-orange-400 rounded-full border border-orange-200 dark:border-orange-700 leading-none">BETA</span>
                  </h1>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={toggleTheme}
                  className="p-2 text-gray-600 dark:text-gray-400 hover:text-orange-600 dark:hover:text-orange-400 hover:bg-orange-50 dark:hover:bg-gray-800 rounded-xl transition-all shadow-sm border border-orange-50 dark:border-gray-700 bg-white dark:bg-gray-800"
                  aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
                >
                  {theme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
                </button>
                <button
                  onClick={() => {
                    if (messages.length > 0 && !window.confirm('Start a new conversation? Your current conversation is saved in history.')) return;
                    handleNewConversation();
                  }}
                  className="flex items-center gap-1.5 px-4 py-2 text-xs font-black text-orange-600 dark:text-orange-400 hover:bg-orange-50 dark:hover:bg-gray-800 border border-orange-100 dark:border-gray-700 rounded-xl transition-all shadow-sm bg-white dark:bg-gray-800 active:scale-95"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                  <span className="hidden sm:inline uppercase">New Session</span>
                </button>
              </div>
            </div>
          </header>

          <div className="flex-1 relative overflow-hidden flex flex-col">
            <div ref={scrollContainerRef} className="flex-1 overflow-y-auto pt-4 pb-32 scroll-smooth scrollbar-hide">
              <div className="max-w-4xl mx-auto px-5">
                {messages.length === 0 ? (
                  <div className="text-center py-12 md:py-16 animate-fade-in px-4">
                    <div className="mb-8 md:mb-12 relative inline-block">
                      <div className="absolute -inset-10 bg-orange-200/20 blur-[80px] rounded-full -z-10"></div>
                      <div className="flex flex-col items-center">
                        <img src={theme === 'dark' ? '/logo-full-dark.png' : '/logo-full.png'} alt="3ioNetra" className="h-16 md:h-20 lg:h-24 object-contain mb-4 dark:drop-shadow-[0_0_15px_rgba(255,255,255,0.12)]" />
                        <div className="h-1 w-48 bg-gradient-to-r from-orange-400 via-amber-500 to-orange-400 rounded-full shadow shadow-orange-100"></div>
                      </div>
                      <p className="text-[10px] md:text-xs text-orange-600 font-black uppercase tracking-[0.3em] mt-3">{greeting}</p>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 md:gap-6 mt-8 md:mt-12 max-w-2xl mx-auto">
                      <button
                        type="button"
                        onClick={() => setInput("I'd like spiritual guidance from the scriptures")}
                        className="p-6 md:p-8 bg-white dark:bg-gray-800 border border-gray-100 dark:border-gray-700 rounded-3xl shadow-sm hover:shadow-xl hover:-translate-y-1 transition-all text-left group cursor-pointer active:scale-[0.98] focus:outline-none focus:ring-2 focus:ring-orange-500/30"
                      >
                        <div className="w-10 h-10 md:w-12 md:h-12 bg-orange-50 dark:bg-orange-900/30 rounded-2xl flex items-center justify-center mb-4 md:mb-6 text-orange-600 group-hover:scale-110 group-hover:bg-orange-600 group-hover:text-white transition-all duration-500 shadow-inner">
                          <BookOpen className="w-5 h-5 md:w-6 md:h-6" />
                        </div>
                        <h3 className="font-black text-gray-900 dark:text-gray-100 mb-1.5 text-lg md:text-xl tracking-tight">Seek Wisdom</h3>
                        <p className="text-xs md:text-sm text-gray-500 dark:text-gray-400 leading-relaxed font-medium">Explore Sanatan Dharma through sacred scriptures and tailored guidance.</p>
                      </button>
                      <button
                        type="button"
                        onClick={() => setInput("I'm going through a difficult time and need guidance")}
                        className="p-6 md:p-8 bg-white dark:bg-gray-800 border border-gray-100 dark:border-gray-700 rounded-3xl shadow-sm hover:shadow-xl hover:-translate-y-1 transition-all text-left group cursor-pointer active:scale-[0.98] focus:outline-none focus:ring-2 focus:ring-orange-500/30"
                      >
                        <div className="w-10 h-10 md:w-12 md:h-12 bg-amber-50 dark:bg-amber-900/30 rounded-2xl flex items-center justify-center mb-4 md:mb-6 text-amber-600 group-hover:scale-110 group-hover:bg-amber-600 group-hover:text-white transition-all duration-500 shadow-inner">
                          <Activity className="w-5 h-5 md:w-6 md:h-6" />
                        </div>
                        <h3 className="font-black text-gray-900 dark:text-gray-100 mb-1.5 text-lg md:text-xl tracking-tight">Daily Support</h3>
                        <p className="text-xs md:text-sm text-gray-500 dark:text-gray-400 leading-relaxed font-medium">Share your life's challenges and find spiritual peace and direction.</p>
                      </button>
                    </div>
                  </div>
                ) : (
                  <div role="log" aria-label="Conversation with Mitra" aria-live="polite" className="space-y-6 md:space-y-8 pb-10">
                    {messages.map((message, index) => {
                      if (message.role === 'assistant' && !message.content) return null;
                      return (
                      <div key={index} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} animate-fade-in`}>
                        <div className={`max-w-[85%] sm:max-w-[70%] lg:max-w-[680px] rounded-2xl px-4 py-3 md:px-6 md:py-4 shadow-xl shadow-orange-900/[0.03] dark:shadow-black/10 transition-all relative group ${message.role === 'user'
                          ? 'bg-gradient-to-br from-orange-500 to-amber-600 text-white shadow-orange-200/30 rounded-tr-sm'
                          : 'bg-white dark:bg-gray-800 border border-orange-50/50 dark:border-gray-700 text-gray-800 dark:text-gray-200 rounded-tl-sm shadow-sm'
                          }`}>
                          {message.role === 'assistant' && message.content ? (() => {
                            if (message.isError) {
                              return (
                                <div className="space-y-2 text-sm md:text-[15px]">
                                  <p className="whitespace-pre-wrap leading-[1.65] font-medium text-gray-700/90 dark:text-gray-300">{message.content}</p>
                                  <button
                                    onClick={() => handleRetry(index)}
                                    className="flex items-center gap-1.5 px-3 py-1.5 text-[10px] font-black uppercase tracking-wider text-orange-600 dark:text-orange-400 border border-orange-200 dark:border-orange-800 rounded-full hover:bg-orange-50 dark:hover:bg-orange-900/20 transition-all active:scale-95"
                                  >
                                    <RefreshCw className="w-3 h-3" />
                                    Try again
                                  </button>
                                </div>
                              );
                            }
                            const segments = (index < parsedMessages.length ? parsedMessages[index] : null) || parseResponseForVerses(message.content);
                            const textAndVerseSegs = segments.filter(s => s.type !== 'mantra');
                            const mantraSegs = segments.filter(s => s.type === 'mantra');
                            return (
                              <div className="message-content space-y-2.5 md:space-y-3 text-sm md:text-[15px]">
                                {/* Text + Verse segments */}
                                {textAndVerseSegs.map((seg, si) => (
                                  seg.type === 'verse' ? (
                                    <blockquote key={si} className="my-3 md:my-4 pl-3.5 md:pl-5 border-l-4 border-amber-400 dark:border-amber-600 bg-amber-50/40 dark:bg-amber-900/20 rounded-r-xl py-3 md:py-4 pr-3.5 md:pr-5 border border-orange-100/30 dark:border-amber-800/30 relative overflow-hidden">
                                      <p className="whitespace-pre-wrap text-amber-950 dark:text-amber-200 italic text-[14px] leading-relaxed font-serif relative z-10">
                                        &ldquo;{seg.content}&rdquo;
                                      </p>
                                      <div className="mt-2.5 relative z-10 flex items-center gap-2">
                                        <TTSButton text={seg.content} lang="hi" variant="verse" />
                                        <button
                                          onClick={() => handleCopy(seg.content, `verse-${index}-${si}`)}
                                          aria-label="Copy verse text"
                                          className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-[9px] font-black uppercase tracking-wider border border-amber-100 dark:border-amber-800 text-amber-700 dark:text-amber-400 bg-white dark:bg-gray-800 hover:bg-amber-50 dark:hover:bg-amber-900/30 transition-all"
                                        >
                                          {copiedId === `verse-${index}-${si}` ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                                          <span>{copiedId === `verse-${index}-${si}` ? 'Copied!' : 'Copy'}</span>
                                        </button>
                                      </div>
                                    </blockquote>
                                  ) : (
                                    <div key={si} className="prose prose-sm md:prose-base dark:prose-invert max-w-none prose-p:leading-[1.7] prose-p:my-1.5 prose-strong:text-orange-700 dark:prose-strong:text-amber-400 prose-li:my-0.5 prose-ol:my-2 prose-ul:my-2">
                                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{fixInlineBullets(seg.content)}</ReactMarkdown>
                                      {isProcessing && index === messages.length - 1 && si === textAndVerseSegs.length - 1 && seg.type === 'text' && (
                                        <span className="streaming-cursor" />
                                      )}
                                    </div>
                                  )
                                ))}
                                {/* Mantra section — separate dedicated block */}
                                {mantraSegs.length > 0 && (
                                  <div className="mt-4 space-y-3">
                                    {mantraSegs.map((seg, mi) => (
                                      <div key={`mantra-${mi}`} className="rounded-xl bg-gradient-to-r from-amber-50 to-orange-50 dark:from-amber-950/40 dark:to-orange-950/30 border border-amber-200/60 dark:border-amber-800/40 p-4 md:p-5">
                                        <div className="flex items-center gap-2 mb-2.5">
                                          <span className="text-amber-600 dark:text-amber-400 text-lg">🙏</span>
                                          <span className="text-[10px] font-black uppercase tracking-widest text-amber-700 dark:text-amber-400">Recite this Mantra</span>
                                        </div>
                                        <p className="text-center text-lg md:text-xl font-serif italic text-amber-900 dark:text-amber-200 leading-relaxed py-2">
                                          {seg.content}
                                        </p>
                                        <div className="mt-3 flex items-center justify-center gap-3">
                                          <TTSButton text={seg.content} lang="hi" variant="mantra" />
                                          <button
                                            onClick={() => handleCopy(seg.content, `mantra-${index}-${mi}`)}
                                            className="inline-flex items-center gap-1 px-3 py-1.5 rounded-full text-[9px] font-black uppercase tracking-wider border border-amber-200 dark:border-amber-700 text-amber-700 dark:text-amber-400 bg-white/80 dark:bg-gray-800/80 hover:bg-amber-50 dark:hover:bg-amber-900/30 transition-all"
                                          >
                                            {copiedId === `mantra-${index}-${mi}` ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                                            <span>{copiedId === `mantra-${index}-${mi}` ? 'Copied!' : 'Copy'}</span>
                                          </button>
                                        </div>
                                      </div>
                                    ))}
                                  </div>
                                )}
                                <div className="pt-3 border-t border-orange-50/50 dark:border-gray-700 flex items-center justify-between">
                                  <div className="flex items-center gap-1">
                                    <button
                                      onClick={() => handleFeedback(index, message.content, 'like')}
                                      className={`p-1.5 rounded-lg transition-all ${feedback[index] === 'like' ? 'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400' : 'text-gray-500 hover:text-green-500 hover:bg-green-50 dark:hover:bg-green-900/20'}`}
                                      aria-label="Helpful"
                                    >
                                      <ThumbsUp className="w-3.5 h-3.5" />
                                    </button>
                                    <button
                                      onClick={() => handleFeedback(index, message.content, 'dislike')}
                                      className={`p-1.5 rounded-lg transition-all ${feedback[index] === 'dislike' ? 'bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400' : 'text-gray-500 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20'}`}
                                      aria-label="Not helpful"
                                    >
                                      <ThumbsDown className="w-3.5 h-3.5" />
                                    </button>
                                  </div>
                                  <TTSButton text={message.content} lang="en" variant="response" label="Listen to Full Response" />
                                </div>
                                {(() => {
                                  const allCites = [...(message.citations || []), ...(message.sources || [])];
                                  if (allCites.length === 0) return null;
                                  return (
                                  <div className="pt-2">
                                    <button
                                      onClick={() => setExpandedCitations(prev => ({ ...prev, [index]: !prev[index] }))}
                                      aria-expanded={!!expandedCitations[index]}
                                      aria-controls={`cite-list-${index}`}
                                      className="flex items-center gap-1.5 text-[9px] font-black uppercase tracking-wider text-gray-400 dark:text-gray-400 hover:text-orange-600 dark:hover:text-orange-400 transition-colors"
                                    >
                                      <BookOpen className="w-3 h-3" />
                                      <span>
                                        Sources: {allCites.slice(0, 3).map(c => `${'scripture' in c ? c.scripture : ''} ${'reference' in c ? c.reference : ''}`).join(', ')}
                                      </span>
                                      <ChevronDown className={`w-3 h-3 transition-transform ${expandedCitations[index] ? 'rotate-180' : ''}`} />
                                    </button>
                                    <div
                                      id={`cite-list-${index}`}
                                      className={`overflow-hidden transition-all duration-200 ease-out ${expandedCitations[index] ? 'max-h-96 opacity-100 mt-2' : 'max-h-0 opacity-0'}`}
                                    >
                                      <div className="space-y-2">
                                        {allCites.map((cite, ci) => (
                                          <div key={ci} className="pl-3 border-l-2 border-amber-300 dark:border-amber-700 py-1.5">
                                            <p className="text-[10px] font-black text-orange-700 dark:text-orange-400 uppercase tracking-wider">
                                              {'scripture' in cite ? cite.scripture : ''} {'reference' in cite ? cite.reference : ''}
                                            </p>
                                            <p className="text-[11px] text-gray-600 dark:text-gray-400 leading-relaxed mt-0.5">
                                              {'text' in cite ? cite.text : 'context_text' in cite ? cite.context_text : ''}
                                            </p>
                                          </div>
                                        ))}
                                      </div>
                                    </div>
                                  </div>
                                  );
                                })()}
                              </div>
                            );
                          })() : (
                            <p className="whitespace-pre-wrap text-[14px] md:text-[15px] leading-[1.65] md:leading-[1.7] font-bold">
                              {message.content}
                            </p>
                          )}



                          {message.role === 'assistant' && message.recommendedProducts && message.recommendedProducts.length > 0 && (
                            <ProductDisplay products={message.recommendedProducts} />
                          )}

                          <div className={`mt-2 text-[8px] md:text-[9px] font-black tracking-widest uppercase opacity-30 ${message.role === 'user' ? 'text-white' : 'text-gray-400'}`}>
                            {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                          </div>
                        </div>
                      </div>
                      );
                    })}

                    {isProcessing && (!messages.length || messages[messages.length - 1]?.content === '') && (
                      <div className="flex justify-start animate-fade-in pl-1" role="status" aria-live="polite">
                        <div className="bg-white/90 dark:bg-gray-800/90 backdrop-blur-xl border border-orange-100 dark:border-gray-700 shadow-lg rounded-xl px-5 py-3 flex items-center gap-4">
                          <div className="flex gap-1">
                            <div className="w-1 h-1 bg-orange-400 dark:bg-orange-500 rounded-full animate-bounce [animation-duration:0.8s]"></div>
                            <div className="w-1 h-1 bg-orange-500 dark:bg-orange-400 rounded-full animate-bounce [animation-delay:0.2s] [animation-duration:0.8s]"></div>
                            <div className="w-1 h-1 bg-orange-600 dark:bg-orange-300 rounded-full animate-bounce [animation-delay:0.4s] [animation-duration:0.8s]"></div>
                          </div>
                          <span className="text-[9px] font-black text-orange-900 dark:text-orange-300 uppercase tracking-widest">
                            {loadingStatus || (
                              session.phase === 'synthesis' ? 'Mitra is reflecting...' :
                              session.phase === 'guidance' || session.phase === 'answering' ? 'Mitra is composing guidance...' :
                              'Mitra is listening...'
                            )}
                          </span>
                        </div>
                      </div>
                    )}
                    <div ref={messagesEndRef} />
                  </div>
                )}
              </div>
            </div>

            {/* Scroll-to-bottom FAB */}
            {showScrollBottom && messages.length > 0 && (
              <button
                onClick={scrollToBottom}
                className="absolute bottom-36 right-6 z-20 p-2.5 bg-white dark:bg-gray-800 border border-orange-200 dark:border-gray-600 rounded-full shadow-lg hover:shadow-xl hover:bg-orange-50 dark:hover:bg-gray-700 transition-all animate-fade-in"
                aria-label="Scroll to bottom"
              >
                <ArrowDown className="w-4 h-4 text-orange-600 dark:text-orange-400" />
              </button>
            )}

            <div className="absolute bottom-0 left-0 right-0 p-4 md:p-6 pointer-events-none">
              <div className="max-w-3xl mx-auto pointer-events-auto">
                {messages.length === 0 && (
                  <div className="flex flex-wrap gap-2 justify-center mb-3">
                    {['I\'m feeling anxious', 'Guide me with a mantra', 'Help with a life decision', 'Today\'s panchang'].map((text) => (
                      <button
                        key={text}
                        onClick={() => { setInput(text); textareaRef.current?.focus(); }}
                        className="text-[11px] font-bold px-3 py-1.5 rounded-full border border-orange-200 dark:border-orange-800 text-orange-700 dark:text-orange-300 hover:bg-orange-50 dark:hover:bg-orange-900/20 bg-white dark:bg-gray-800 transition-all active:scale-95 shadow-sm"
                      >
                        {text}
                      </button>
                    ))}
                  </div>
                )}
                <div className="bg-white/90 dark:bg-gray-900/90 backdrop-blur-[12px] border border-orange-100 dark:border-gray-700 shadow-xl rounded-2xl p-1 pr-2.5 flex items-end gap-1 group transition-all duration-700 hover:shadow-orange-900/5 focus-within:ring-[6px] focus-within:ring-orange-500/20">
                  <form onSubmit={handleTextSubmit} className="flex-1 flex items-end">
                    <label htmlFor="chat-input" className="sr-only">Send a message to Mitra</label>
                    <textarea
                      ref={textareaRef}
                      id="chat-input"
                      data-testid="chat-input"
                      name="message"
                      value={input}
                      onChange={(e) => {
                        setInput(e.target.value);
                        // Auto-resize
                        e.target.style.height = 'auto';
                        e.target.style.height = Math.min(e.target.scrollHeight, 140) + 'px';
                      }}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                          e.preventDefault();
                          if (input.trim() && !isProcessing) {
                            handleTextSubmit(e as unknown as React.FormEvent);
                          }
                        }
                      }}
                      placeholder="Share your spiritual journey..."
                      disabled={isProcessing}
                      autoComplete="off"
                      rows={1}
                      className="flex-1 px-4 md:px-6 py-3 md:py-4 bg-transparent text-gray-950 dark:text-gray-100 placeholder:text-gray-400 dark:placeholder:text-gray-500 border-none focus:ring-0 text-[14px] md:text-[15px] font-bold tracking-tight outline-none resize-none max-h-[140px]"
                    />
                    <button
                      type="submit"
                      data-testid="send-button"
                      aria-label={isProcessing ? "Sending message..." : "Send message"}
                      disabled={isProcessing || !input.trim()}
                      className="p-3 md:p-3.5 mb-1 bg-gradient-to-br from-orange-500 to-amber-700 hover:from-orange-600 hover:to-amber-800 text-white rounded-xl shadow-lg disabled:grayscale disabled:opacity-20 transition-all duration-500 active:scale-95"
                    >
                      {isProcessing ? <Loader2 className="w-4.5 h-4.5 animate-spin" /> : <Send className="w-4.5 h-4.5" />}
                    </button>
                  </form>
                </div>
                <p className="text-center text-[9px] text-gray-400 dark:text-gray-600 mt-1.5 font-medium">
                  Press Enter to send, Shift+Enter for new line
                </p>
              </div>
            </div>
          </div>
        </div>
      </main>
    </>
  );
}
