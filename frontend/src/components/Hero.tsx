export const Hero = () => {
  return (
    <header className="mb-16">
      <div className="flex justify-between items-start mb-8 border-b-2 border-hhgoa-border pb-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight">HHGOA'26</h1>
          <h2 className="text-md font-medium text-gray-600">VOICE RAG</h2>
        </div>
        <div className="text-right">
          <p className="text-xs uppercase tracking-widest font-bold">VOICE → RETRIEVE → ANSWER</p>
          <p className="text-xs uppercase tracking-widest font-bold mt-1 text-hhgoa-brand">AI4BHARAT / MSMARCO-XI</p>
        </div>
      </div>
      
      <div className="max-w-4xl">
        <h1 className="text-6xl md:text-8xl font-black tracking-tighter uppercase leading-none mb-6">
          Ask The<br />Corpus.
        </h1>
        <p className="text-lg md:text-xl font-medium max-w-2xl text-gray-800 leading-relaxed border-l-4 border-hhgoa-brand pl-4">
          Speak a question. Retrieve evidence. Get an answer grounded in the provided knowledge base.
        </p>
      </div>
    </header>
  );
};
