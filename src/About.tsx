import { type Word, type Keyword } from './nodes/types';
import { TextWithKeywordsNode } from './nodes/TextWithKeywordsNode'; // Adjust the import path as necessary

const About = () => {
  const data = {
    words: [
      { value: 'This', id: '1' },
      { value: 'is' },
      { value: 'a' },
      { value: 'test', id: '4' },
      { value: 'sentence', id: '5' },
    ],
  };

  // const handleKeywordClick = (keyword: Keyword) => {
  //   alert(`Keyword clicked: ${keyword.value}`);
  // };

  return (
    <div>
      <div className="flex h-screen">
        {/* Main Content */}
        <div className="flex-1 p-4 overflow-y-auto">
          <h1 className="text-4xl font-bold text-center mb-8">ArtiFactor</h1>

          {/* Section 1 */}
          <div id="section1" className="mb-8">
            <h3 className="text-2xl font-semibold mb-2">Our Project</h3>
            <p>Pretend we filled this out.</p>
          </div>

          {/* Example Section */}
          <div id="exampleSection" className="mb-8">
            <h3 className="text-2xl font-semibold mb-2">Example Components</h3>
            <TextWithKeywordsNode data={data}/>
          </div>
        </div>
      </div>
    </div>
  );
};

export default About;