<script lang="ts">
  import { marked } from 'marked';

  interface Props {
    content: string;
    class?: string;
  }

  let { content, class: className = '' }: Props = $props();

  // Create a custom renderer to match the React component's styling
  const renderer = new marked.Renderer();

  // Map classes from the React reference
  renderer.heading = ({ text, depth }) => {
    const sizes = {
      1: "text-2xl font-semibold",
      2: "font-semibold text-xl",
      3: "font-semibold text-lg",
      4: "font-semibold text-base",
      5: "font-medium",
      6: "font-medium"
    };
    const sizeClass = sizes[depth as keyof typeof sizes] || sizes[6];
    return `<h${depth} class="${sizeClass} mt-4 mb-2">${text}</h${depth}>`;
  };

  renderer.paragraph = ({ text }) => {
    return `<p class="whitespace-pre-wrap my-2 mb-4 leading-relaxed">${text}</p>`;
  };

  renderer.strong = ({ text }) => {
    return `<strong class="font-semibold">${text}</strong>`;
  };

  renderer.link = ({ href, title, text }) => {
    return `<a href="${href}" class="text-blue-600 underline underline-offset-2 hover:text-blue-500" title="${title || ''}">${text}</a>`;
  };

  renderer.blockquote = ({ text }) => {
    return `<blockquote class="border-l-2 border-blue-600 pl-4 my-4 italic text-gray-700 bg-gray-50 py-2 pr-2 rounded-r">${text}</blockquote>`;
  };

  renderer.list = ({ body, ordered }) => {
    const type = ordered ? "ol" : "ul";
    const style = ordered ? "list-decimal space-y-2 pl-6" : "list-disc space-y-2 pl-6";
    return `<${type} class="${style} my-4">${body}</${type}>`;
  };

  renderer.listitem = ({ text }) => {
    return `<li class="my-1.5 pl-1">${text}</li>`;
  };

  renderer.codespan = ({ text }) => {
    return `<code class="font-mono bg-gray-100 px-1.5 py-0.5 rounded text-sm text-gray-800">${text}</code>`;
  };

  renderer.code = ({ text, lang }) => {
    return `<pre class="overflow-x-auto rounded-md border border-gray-200 bg-gray-50 p-4 font-mono text-sm my-4"><code class="language-${lang || 'text'}">${text}</code></pre>`;
  };

  renderer.hr = () => {
    return `<hr class="border-gray-200 my-6" />`;
  };

  renderer.table = ({ header, body }) => {
    return `
      <div class="overflow-x-auto my-4 rounded-md border border-gray-200">
        <table class="w-full border-collapse text-sm">
          <thead class="bg-gray-50 border-b border-gray-200">
            ${header}
          </thead>
          <tbody class="divide-y divide-gray-100">
            ${body}
          </tbody>
        </table>
      </div>
    `;
  };

  renderer.tablerow = ({ content }) => {
    return `<tr class="hover:bg-gray-50/50 transition-colors">${content}</tr>`;
  };

  renderer.tablecell = ({ content, flags: { header, align } }) => {
    const tag = header ? "th" : "td";
    const alignClass = align ? `text-${align}` : "text-left";
    const baseClass = header 
      ? "px-4 py-3 font-semibold text-gray-700" 
      : "px-4 py-3 text-gray-600";
    return `<${tag} class="${baseClass} ${alignClass}">${content}</${tag}>`;
  };

  // Configure marked
  marked.use({ 
    renderer,
    breaks: false, // Important for tables
    gfm: true 
  });

  // Preprocessing functions from the reference + table fix
  function normalizeMathDelimiters(text: string): string {
    text = text.replace(/\\\((.+?)\\\)/g, (_m, inner) => `$${inner}$`.replace(/\$\$/g, '$'));
    text = text.replace(/^\s*\\\[\s*$/gm, '$$');
    text = text.replace(/^\s*\\\]\s*$/gm, '$$');
    return text;
  }

  function fixBrokenInlineDollarMathInTables(text: string): string {
    return text.replace(
      /(\|\s*)\$\s*\n([^$]*?)\n\$\s*(\|)/g,
      (_match, left, inner, right) => `${left}$${inner.replace(/\n/g, ' ')}$${right}`
    );
  }

  function flattenInlineMathNewlines(text: string): string {
    let result = '';
    let inInlineMath = false;
    let inDisplayMath = false;
    let inCodeFence = false;

    for (let i = 0; i < text.length; i++) {
      if (text.startsWith('```', i)) {
        inCodeFence = !inCodeFence;
        result += '```';
        i += 2;
        continue;
      }
      const ch = text[i];
      if (inCodeFence) { result += ch; continue; }
      if (ch === '$') {
        const nextIsDollar = text[i + 1] === '$';
        if (nextIsDollar) { inDisplayMath = !inDisplayMath; result += '$$'; i++; continue; }
        else { inInlineMath = !inInlineMath; result += '$'; continue; }
      }
      if (ch === '\n' && inInlineMath && !inDisplayMath) { result += ' '; } 
      else { result += ch; }
    }
    return result;
  }

  function stripCitations(text: string): string {
    return text.replace(/【[^】]*†[^】]*】/g, '');
  }

  function fixMalformedTables(text: string): string {
    let processed = text;
    // Fix merged header: "Sandwiches| Item | Price |" -> "## Sandwiches\n\n| Item | Price |"
    processed = processed.replace(
      /^([A-Za-z][A-Za-z\s]*?)\|\s*([A-Za-z]+)\s*\|\s*([A-Za-z]+)\s*\|/gm,
      (match, category, col1, col2) => {
        const cat = category.trim();
        if (cat.startsWith('#') || cat.startsWith('|')) return match;
        return `## ${cat}\n\n| ${col1} | ${col2} |`;
      }
    );
    // Fix multiple rows on one line
    processed = processed.replace(/\|\s*\|\s*([^|\n]+)\s*\|/g, '|\n| $1 |');
    
    // Fix missing separator rows
    const lines = processed.split('\n');
    const fixedLines: string[] = [];
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      fixedLines.push(line);
      // More robust check for table header
      if (line.match(/^\|\s*[A-Za-z]+\s*\|/)) {
        const nextLine = lines[i + 1];
        if (!nextLine || !nextLine.match(/^\s*\|[-:\s|]+\|\s*$/)) {
          const cols = (line.match(/\|/g) || []).length - 1;
          if (cols > 0) {
            fixedLines.push('|' + Array(cols).fill('------|').join(''));
          }
        }
      }
    }
    return fixedLines.join('\n');
  }

  const renderMarkdown = (text: string): string => {
    try {
      let processed = text;
      processed = stripCitations(processed);
      processed = fixMalformedTables(processed); // Run this EARLY
      processed = normalizeMathDelimiters(processed);
      processed = fixBrokenInlineDollarMathInTables(processed);
      processed = flattenInlineMathNewlines(processed);

      // marked.parse returns string when async: false (default)
      const result = marked.parse(processed, { async: false });
      return result as string;
    } catch (e) {
      console.error('Markdown parse error:', e);
      return text;
    }
  };

  let html = $derived(renderMarkdown(content));
</script>

<div class={className}>
  {@html html}
</div>
