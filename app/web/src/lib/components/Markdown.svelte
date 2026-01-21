<script lang="ts">
  import MarkdownIt from 'markdown-it';

  interface Props {
    content: string;
    class?: string;
  }

  let { content, class: className = '' }: Props = $props();

  // Create markdown-it instance with GFM-like settings
  const md = new MarkdownIt({
    html: false,
    breaks: false,
    linkify: true,
    typographer: false,
  }).enable('table'); // Enable GFM tables

  // Preprocessing functions
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
    
    // Remove lines that are only dashes/tabs/spaces (malformed separators between data rows)
    // These look like: "------\t------" or "------|------" etc.
    processed = processed.replace(/^[\s\t]*-{2,}[\s\t|]*-{2,}[\s\t]*$/gm, '');
    
    // Clean up multiple empty lines that result from removing separators
    processed = processed.replace(/\n{3,}/g, '\n\n');
    
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
    
    // Ensure tables have proper separator row after header
    const lines = processed.split('\n');
    const fixedLines: string[] = [];
    let inTable = false;
    let hadSeparator = false;
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const isSeparatorRow = /^[\s|:-]+$/.test(line) && line.includes('-') && line.includes('|');
      const isTableRow = line.trim().startsWith('|') && line.trim().endsWith('|');
      
      if (isTableRow && !inTable) {
        inTable = true;
        hadSeparator = false;
        fixedLines.push(line);
        
        // Check if next line is a separator, if not add one
        const nextLine = lines[i + 1];
        if (!nextLine || !/^[\s|:-]+$/.test(nextLine) || !nextLine.includes('-')) {
          const cols = (line.match(/\|/g) || []).length - 1;
          if (cols > 0) {
            fixedLines.push('|' + Array(cols).fill(' --- |').join(''));
            hadSeparator = true;
          }
        }
        continue;
      }
      
      if (inTable) {
        if (isSeparatorRow) {
          if (!hadSeparator) {
            fixedLines.push(line);
            hadSeparator = true;
          }
          continue;
        } else if (isTableRow) {
          fixedLines.push(line);
        } else if (line.trim() === '') {
          inTable = false;
          hadSeparator = false;
          fixedLines.push(line);
        } else {
          inTable = false;
          hadSeparator = false;
          fixedLines.push(line);
        }
      } else {
        fixedLines.push(line);
      }
    }
    return fixedLines.join('\n');
  }

  const renderMarkdown = (text: string): string => {
    try {
      let processed = text;
      processed = stripCitations(processed);
      processed = fixMalformedTables(processed);
      processed = normalizeMathDelimiters(processed);
      processed = fixBrokenInlineDollarMathInTables(processed);
      processed = flattenInlineMathNewlines(processed);

      return md.render(processed);
    } catch (e) {
      console.error('Markdown parse error:', e);
      return text;
    }
  };

  let html = $derived(renderMarkdown(content));
</script>

<div class="markdown-content {className}">
  {@html html}
</div>

<style>
  .markdown-content :global(h1) {
    font-size: 1.5rem;
    font-weight: 600;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
  }
  
  .markdown-content :global(h2) {
    font-size: 1.25rem;
    font-weight: 600;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
  }
  
  .markdown-content :global(h3) {
    font-size: 1.125rem;
    font-weight: 600;
    margin-top: 0.75rem;
    margin-bottom: 0.5rem;
  }
  
  .markdown-content :global(p) {
    white-space: pre-wrap;
    margin: 0.5rem 0 1rem 0;
    line-height: 1.625;
  }
  
  .markdown-content :global(strong) {
    font-weight: 600;
  }
  
  .markdown-content :global(a) {
    color: #2563eb;
    text-decoration: underline;
    text-underline-offset: 2px;
  }
  
  .markdown-content :global(a:hover) {
    color: #3b82f6;
  }
  
  .markdown-content :global(blockquote) {
    border-left: 2px solid #2563eb;
    padding-left: 1rem;
    margin: 1rem 0;
    font-style: italic;
    color: #374151;
    background-color: #f9fafb;
    padding-top: 0.5rem;
    padding-bottom: 0.5rem;
    padding-right: 0.5rem;
    border-radius: 0 0.25rem 0.25rem 0;
  }
  
  .markdown-content :global(ul) {
    list-style-type: disc;
    padding-left: 1.5rem;
    margin: 1rem 0;
  }
  
  .markdown-content :global(ol) {
    list-style-type: decimal;
    padding-left: 1.5rem;
    margin: 1rem 0;
  }
  
  .markdown-content :global(li) {
    margin: 0.375rem 0;
    padding-left: 0.25rem;
  }
  
  .markdown-content :global(code) {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    background-color: #f3f4f6;
    padding: 0.125rem 0.375rem;
    border-radius: 0.25rem;
    font-size: 0.875rem;
    color: #1f2937;
  }
  
  .markdown-content :global(pre) {
    overflow-x: auto;
    border-radius: 0.375rem;
    border: 1px solid #e5e7eb;
    background-color: #f9fafb;
    padding: 1rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.875rem;
    margin: 1rem 0;
  }
  
  .markdown-content :global(pre code) {
    background-color: transparent;
    padding: 0;
    border-radius: 0;
  }
  
  .markdown-content :global(hr) {
    border-color: #e5e7eb;
    margin: 1.5rem 0;
  }
  
  /* Table styles */
  .markdown-content :global(table) {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.875rem;
    margin: 1rem 0;
    border: 1px solid #e5e7eb;
    border-radius: 0.375rem;
    overflow: hidden;
  }
  
  .markdown-content :global(thead) {
    background-color: #f9fafb;
    border-bottom: 1px solid #e5e7eb;
  }
  
  .markdown-content :global(tbody) {
    /* divide-y equivalent */
  }
  
  .markdown-content :global(tr) {
    border-bottom: 1px solid #f3f4f6;
  }
  
  .markdown-content :global(tr:hover) {
    background-color: rgba(249, 250, 251, 0.5);
  }
  
  .markdown-content :global(th) {
    padding: 0.75rem 1rem;
    font-weight: 600;
    color: #374151;
    text-align: left;
  }
  
  .markdown-content :global(td) {
    padding: 0.75rem 1rem;
    color: #4b5563;
  }
</style>
