import { ArrowLeft, ArrowRight } from 'lucide-react';
const NavigationButtons: React.FC<{ currentIndex: number; totalItems: number; handlePrev: () => void; handleNext: () => void }> = ({ currentIndex, totalItems, handlePrev, handleNext }) => {
    return (
        <div className="text-xs/6 flex justify-between mt-2 items-center mx-auto" style={{width: '80%'}}>
            <div onClick={handlePrev} className="p-2 rounded-full text-gray-500 hover:text-amber-800 cursor-pointer">
                <ArrowLeft size='18' />
            </div>
            <span className="text-gray-600">{currentIndex + 1}/{totalItems}</span>
            <div onClick={handleNext} className="p-2 rounded-full text-gray-500 hover:text-amber-800 cursor-pointer">
                <ArrowRight size='18' />
            </div>
        </div>
    );
};

export default NavigationButtons;