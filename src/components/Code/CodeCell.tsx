import React, { createContext, useContext, useRef, useEffect } from "react";
import hljs from "highlight.js/lib/core";
import python from "highlight.js/lib/languages/python";

// регистрируем язык
hljs.registerLanguage("python", python);

type CodeCellContextType = { next: () => number };
const CodeCellContext = createContext<CodeCellContextType | null>(null);

export const CodeCellProvider = ({ children }: { children: React.ReactNode }) => {
  const counterRef = useRef(1);
  return (
    <CodeCellContext.Provider value={{ next: () => counterRef.current++ }}>
      {children}
    </CodeCellContext.Provider>
  );
};

type CodeCellProps = {
  language?: string;
  title?: string;
  children?: React.ReactNode;
  output?: React.ReactNode;
  n?: number;
};

const CodeCell: React.FC<CodeCellProps> = ({
  language = "python",
  title,
  children,
  output,
  n,
}) => {
  const ctx = useContext(CodeCellContext);
  const idx = n !== undefined ? n : (ctx?.next() ?? 0);

  const codeRef = useRef<HTMLElement>(null);

  const hasInput =
    children !== undefined &&
    children !== null &&
    !(typeof children === "string" && children.trim() === "");

  const hasOutput =
    output !== undefined &&
    output !== null &&
    !(typeof output === "string" && output.trim() === "");

  // 🔥 подсветка кода
  useEffect(() => {
    if (codeRef.current) {
      hljs.highlightElement(codeRef.current);
    }
  }, [children]);

  const renderOutput = () => {
    if (!hasOutput) return null;

    // строка
    if (typeof output === "string") {
      const trimmed = output.trim();

      // HTML
      if (trimmed.startsWith("<")) {
        return (
          <div
            className="overflow-x-auto text-sm text-gray-800"
            dangerouslySetInnerHTML={{ __html: output }}
          />
        );
      }

      // обычный текст
      return (
        <pre className="text-xs text-gray-700 whitespace-pre-wrap font-mono">
          {output}
        </pre>
      );
    }

    // JSX / картинка
    return <div className="overflow-x-auto">{output}</div>;
  };

  return (
    <div className="my-6 border border-gray-200 rounded-xl overflow-hidden shadow-sm">

      {/* INPUT */}
      {hasInput && (
        <>
          <div className="flex justify-between items-center px-4 py-2 text-xs font-mono bg-gray-100 border-b border-gray-200">
            <span className="font-semibold text-gray-600" suppressHydrationWarning>
              In [{idx}]:
            </span>
            <span className="uppercase tracking-wide text-gray-400 text-[10px]">
              {title ?? language}
            </span>
          </div>

          <pre className="overflow-x-auto text-sm bg-[#0d1117] text-gray-100 p-4 m-0">
            <code
              ref={codeRef}
              className={`language-${language} font-mono text-[13px]`}
            >
              {children}
            </code>
          </pre>
        </>
      )}

      {/* OUTPUT */}
      {hasOutput && (
        <div className="border-t border-gray-200 bg-white px-4 py-3">
          <span
            className="text-xs font-mono font-semibold text-gray-500 block mb-2"
            suppressHydrationWarning
          >
            Out [{idx}]:
          </span>
          {renderOutput()}
        </div>
      )}
    </div>
  );
};

export default CodeCell;