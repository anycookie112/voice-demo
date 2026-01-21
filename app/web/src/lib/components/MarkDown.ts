'use client';

import React, { Suspense, useEffect, useState } from "react"
import Markdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { cn } from "@/lib/utils"
import { CopyButton } from "@/components/ui/copy-button"
import rehypeRaw from "rehype-raw";
import remarkMath from "remark-math"
import rehypeKatex from "rehype-katex"
import "katex/dist/katex.min.css" 

function normalizeMathDelimiters(text) {
  // Inline: \( ... \)  →  $...$
  // ('.' does NOT cross newlines, so it stays within one line)
  text = text.replace(/\\\((.+?)\\\)/g, (_m, inner) =>
    `$${inner}$`.replace(/\$\$/g, "$")
  );
  // Block open: a line that is ONLY `\[`  →  `$$`
  text = text.replace(/^\s*\\\[\s*$/gm, "$$");
  // Block close: a line that is ONLY `\]`  →  `$$`
  text = text.replace(/^\s*\\\]\s*$/gm, "$$");
  return text;
}

function fixBrokenInlineDollarMathInTables(text) {
  // Fix pattern:
  //   | ... | $
  //   <stuff without $>
  //   $ |
  // →  | ... | $<stuff>$ |
  return text.replace(
    /(\|\s*)\$\s*\n([^$]*?)\n\$\s*(\|)/g,
    (_match, left, inner, right) =>
      `${left}$${inner.replace(/\n/g, " ")}$${right}`
  );
}

function flattenInlineMathNewlines(text) {
  let result = ""
  let inInlineMath = false
  let inDisplayMath = false
  let inCodeFence = false

  for (let i = 0; i < text.length; i++) {
    // Handle fenced code blocks ```...```
    if (text.startsWith("```", i)) {
      inCodeFence = !inCodeFence
      result += "```"
      i += 2 // skip the next two backticks
      continue
    }

    const ch = text[i]

    if (inCodeFence) {
      // Do not touch anything inside code fences
      result += ch
      continue
    }

    if (ch === "$") {
      // Check for $$ (display math)
      const nextIsDollar = text[i + 1] === "$"

      if (nextIsDollar) {
        // Toggle display math
        inDisplayMath = !inDisplayMath
        result += "$$"
        i++ // skip second $
        continue
      } else {
        // Toggle inline math
        inInlineMath = !inInlineMath
        result += "$"
        continue
      }
    }

    if (ch === "\n" && inInlineMath && !inDisplayMath) {
      // Replace newline inside $...$ with a space
      result += " "
    } else {
      result += ch
    }
  }
  return result
}

function stripCitations(text) {
  // Removes anything that looks like 【...†...】 (keeps normal [] markdown intact)
  return text.replace(/【[^】]*†[^】]*】/g, "");
}


export function MarkdownRenderer({
  children, isError
}) {
  let normalized = children;

  if(typeof children === "string"){
    normalized = stripCitations(children);

    const withStandardDelimiters = normalizeMathDelimiters(normalized);
    normalized = flattenInlineMathNewlines(withStandardDelimiters);
    // normalized = withStandardDelimiters;
  }

  return (
    <div className={
      cn("space-y-3", isError ? "text-red-500" : "")}
    >
      <Markdown 
        remarkPlugins={[remarkMath, remarkGfm]}
        rehypePlugins={[rehypeRaw, rehypeKatex]} // allow raw HTML like <br>, <b>, etc.
        components={COMPONENTS}
      >
        {normalized}
      </Markdown>
    </div>
  );
}

/* Server Component
const HighlightedPre = React.memo(async ({
  children,
  language,
  ...props
}) => {
  const { codeToTokens, bundledLanguages } = await import("shiki")

  if (!(language in bundledLanguages)) {
    return <pre {...props}>{children}</pre>;
  }

  const { tokens } = await codeToTokens(children, {
    lang: language,
    defaultColor: false,
    themes: {
      light: "github-light",
      dark: "github-dark",
    },
  })

  return (
    <pre {...props}>
      <code>
        {tokens.map((line, lineIndex) => (
          <>
            <span key={lineIndex}>
              {line.map((token, tokenIndex) => {
                const style =
                  typeof token.htmlStyle === "string"
                    ? undefined
                    : token.htmlStyle

                return (
                  <span
                    key={tokenIndex}
                    className="text-shiki-light bg-shiki-light-bg dark:text-shiki-dark dark:bg-shiki-dark-bg"
                    style={style}>
                    {token.content}
                  </span>
                );
              })}
            </span>
            {lineIndex !== tokens.length - 1 && "\n"}
          </>
        ))}
      </code>
    </pre>
  );
})
HighlightedPre.displayName = "HighlightedCode"
*/

// Client component
const HighlightedPre = React.memo(({ children, language, ...props }) => {
  const [tokens, setTokens] = useState(null);

  useEffect(() => {
    (async () => {
      const { codeToTokens, bundledLanguages } = await import('shiki');

      if (!(language in bundledLanguages)) return;

      const { tokens } = await codeToTokens(children, {
        lang: language,
        defaultColor: false,
        themes: {
          light: 'github-light',
          dark: 'github-dark',
        },
      });

      setTokens(tokens);
    })();
  }, [children, language]);

  if (!tokens) return <pre {...props}>{children}</pre>;

  return (
    <pre {...props}>
      <code>
        {tokens.map((line, lineIndex) => (
          <span key={lineIndex}>
            {line.map((token, tokenIndex) => {
              const style =
                typeof token.htmlStyle === 'string' ? undefined : token.htmlStyle;

              return (
                <span
                  key={tokenIndex}
                  className="text-shiki-light bg-shiki-light-bg dark:text-shiki-dark dark:bg-shiki-dark-bg"
                  style={style}>
                  {token.content}
                </span>
              );
            })}
            {lineIndex !== tokens.length - 1 && '\n'}
          </span>
        ))}
      </code>
    </pre>
  );
});

HighlightedPre.displayName = 'HighlightedPre';


const CodeBlock = ({
  children,
  className,
  language,
  ...restProps
}) => {
  const code =
    typeof children === "string"
      ? children
      : childrenTakeAllStringContents(children)

  const preClass = cn(
    "overflow-x-scroll rounded-md border bg-background/50 p-4 font-mono text-sm [scrollbar-width:none] text-wrap",
    className
  )

  return (
    <div className="group/code relative mb-4">
      <Suspense
        fallback={
          <pre className={preClass} {...restProps}>
            {children}
          </pre>
        }>
        <HighlightedPre language={language} className={preClass}>
          {code}
        </HighlightedPre>
      </Suspense>
      <div
        className="invisible absolute right-2 top-2 flex space-x-1 rounded-lg p-1 opacity-0 transition-all duration-200 group-hover/code:visible group-hover/code:opacity-100">
        <CopyButton content={code} copyMessage="Copied code to clipboard" />
      </div>
    </div>
  );
}

function childrenTakeAllStringContents(element) {
  if (typeof element === "string") {
    return element
  }

  if (element?.props?.children) {
    let children = element.props.children

    if (Array.isArray(children)) {
      return children
        .map((child) => childrenTakeAllStringContents(child))
        .join("");
    } else {
      return childrenTakeAllStringContents(children);
    }
  }

  return ""
}

const COMPONENTS = {
  h1: withClass("h1", "text-2xl font-semibold"),
  h2: withClass("h2", "font-semibold text-xl"),
  h3: withClass("h3", "font-semibold text-lg"),
  h4: withClass("h4", "font-semibold text-base"),
  h5: withClass("h5", "font-medium"),
  strong: withClass("strong", "font-semibold"),
  a: withClass("a", "text-primary underline underline-offset-2"),
  blockquote: withClass("blockquote", "border-l-2 border-primary pl-4"),
  code: ({
    children,
    className,
    node,
    ...rest
  }) => {
    const match = /language-(\w+)/.exec(className || "")
    return match ? (
      <CodeBlock className={className} language={match[1]} {...rest}>
        {children}
      </CodeBlock>
    ) : (
      <code
        className={cn(
          "font-mono [:not(pre)>&]:rounded-md [:not(pre)>&]:bg-background/50 [:not(pre)>&]:px-1 [:not(pre)>&]:py-0.5"
        )}
        {...rest}>
        {children}
      </code>
    );
  },
  pre: ({
    children
  }) => children,
  ol: withClass("ol", "list-decimal space-y-2 pl-6"),
  ul: withClass("ul", "list-disc space-y-2 pl-6"),
  li: withClass("li", "my-1.5"),
  table: withClass(
    "table",
    "w-full border-collapse overflow-y-auto rounded-md border border-foreground/20"
  ),
  th: withClass(
    "th",
    "border border-foreground/20 px-4 py-2 text-left font-bold [&[align=center]]:text-center [&[align=right]]:text-right"
  ),
  td: withClass(
    "td",
    "border border-foreground/20 px-4 py-2 text-left [&[align=center]]:text-center [&[align=right]]:text-right"
  ),
  tr: withClass("tr", "m-0 border-t p-0 even:bg-muted"),
  p: withClass("p", "whitespace-pre-wrap"),
  hr: withClass("hr", "border-foreground/20"),
  img: ({ src, alt }) => {
    // If you want to handle local paths (e.g., from /home/...),
    // you might need to rewrite or serve them statically from your backend.
    /*
    const resolvedSrc = src?.startsWith("/home/")
      ? `http://localhost:8000/static/${src.split("/").pop()}` // adjust this base URL
      : src;
    */

    const resolvedSrc = `http://localhost:8000/static/${src.split("/").pop()}`;

    return (
      <img
        src={resolvedSrc}
        alt={alt}
        className="mx-auto my-4 max-h-[400px] w-auto rounded-lg border border-foreground/20 object-contain shadow-sm"
      />
    )
  }
}

function withClass(Tag, classes) {
  const Component = ({
    node,
    ...props
  }) => (
    <Tag className={classes} {...props} />
  )
  Component.displayName = Tag
  return Component
}

export default MarkdownRenderer
