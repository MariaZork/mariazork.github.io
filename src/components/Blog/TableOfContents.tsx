

export type HeadingItem = {
  id: string;
  text: string;
  level: number; // 2, 3, 4
};

interface TableOfContentsProps {
  headings: HeadingItem[];
}

const TableOfContents: React.FC<TableOfContentsProps> = ({ headings }) => {
  if (!headings || headings.length === 0) return null;

  return (
    <nav className="bg-white rounded-lg shadow-lg p-6">
      <h3 className="text-lg font-bold text-dark mb-4">Table of Contents</h3>
      <ul className="space-y-2 text-sm">
        {headings.map((heading) => {
          const indent =
            heading.level === 2 ? '' : heading.level === 3 ? 'ml-3' : 'ml-6';

          return (
            <li key={heading.id} className={indent}>
              <a
                href={`#${heading.id}`}
                className="text-gray-700 hover:text-primary transition"
              >
                {heading.text}
              </a>
            </li>
          );
        })}
      </ul>
    </nav>
  );
};

export default TableOfContents;
