import { ArrowLeft, ArrowRight } from 'lucide-react';

const NavigationButtons: React.FC<{ currentIndex: number; totalItems: number; handlePrev: () => void; handleNext: () => void }> = ({ currentIndex, totalItems, handlePrev, handleNext }) => {
    return (
        <div className="text-xs/6 flex justify-between mt-2 items-center mx-auto" style={{width: '50%'}}>
            <ArrowLeft size='15' onClick={handlePrev} className="text-gray-500 hover:text-amber-800 cursor-pointer" />
            <span className="text-gray-600">{currentIndex + 1}/{totalItems}</span>
            <ArrowRight size='15' onClick={handleNext} className="text-gray-500 hover:text-amber-800 cursor-pointer" />
        </div>
    );
};

export default NavigationButtons;