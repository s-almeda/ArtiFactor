import { motion } from "framer-motion";

export function LoadingNode() {
    return (
        <motion.div
            initial={{ opacity: 0, x: 0, y: 3, scale: 0.6 }}
            animate={{ opacity: 1, x: 0, y: 0, scale: 1, scaleX: 1 }}
            transition={{ duration: 0.4, type: "spring", bounce: 0.1 }}
            className="text-md"
        >
            <div
                style={{
                    display: "flex",
                    flexDirection: "column",
                    justifyContent: "center",
                    alignItems: "center",
                    height: "100%",
                    fontStyle: "italic",
                }}
            >
                <div className="text-gray-800">...loading...</div>
                <div className="loader"></div>
            </div>
        </motion.div>
    );
}