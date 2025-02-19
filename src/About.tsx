// import { type Word, type Keyword } from './nodes/types';
// import { TextWithKeywordsNode } from './nodes/TextWithKeywordsNode'; // Adjust the import path as necessary

const About = () => {

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
            <h3 className="text-2xl font-semibold mb-2">Text with keywords test</h3>
            
          </div>
        </div>
      </div>
    </div>
  );
};

export default About;