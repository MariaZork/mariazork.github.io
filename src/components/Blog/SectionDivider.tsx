export default function SectionDivider() {
  return (
    <div className="my-10 flex items-center justify-center">
      <span className="h-px w-16 bg-gray-300" />
      <span className="mx-3 text-gray-400 text-sm tracking-[0.3em] uppercase">
        • • •
      </span>
      <span className="h-px w-16 bg-gray-300" />
    </div>
  );
}
