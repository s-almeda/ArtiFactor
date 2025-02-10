import React, { createContext, useContext, useState, useEffect } from 'react';

interface AppContextType {
    backend: string;
    //loadCanvasRequest: boolean;
    //setLoadCanvasRequest: React.Dispatch<React.SetStateAction<boolean>>;

    userID: string | null;
    handleUserLogin: (enteredUserID: string, password: string) => void;
    addUser: (userID: string, password: string) => void;

    admins: string[];
}



const AppContext = createContext<AppContextType | undefined>(undefined);

export const AppProvider: React.FC<{ backend: string; children: React.ReactNode }> = ({ backend, children }) => {
    const [userID, setUserID] = useState<string | null>("default");
    const [attemptedQuickLogin, setAttemptedQuickLogin] = useState(false);

    const admins = ["shm", "elaine", "ethan", "sophia"];

    const addUser = async (userID: string, password: string) => {
        try {
            const response = await fetch(`${backend}/api/add-user`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ userID, password })
            });
            const data = await response.json();
            if (response.ok) {
                console.log(`User added: ${userID}`);
            } else {
                console.error(`Error adding user: ${data.error}`);
            }
        } catch (error) {
            console.error('Error adding user:', error);
        }
    };

    const handleUserLogin = async (enteredUserID: string, password?: string) => {
        password = password || "";
        try {
            const response = await fetch(`${backend}/api/list-users`);
            const data = await response.json();
        
            if (response.ok) {
                const userExists = data.users.some((user: { id: string }) => user.id === enteredUserID);
                if (userExists) {
                    setUserID(enteredUserID);
                    console.log(`User logged in: ${enteredUserID}`);
                } else {
                    console.error(`User does not exist: ${enteredUserID}`);
                }
            } else {
                console.error(`Error fetching users: ${data.error}`);
            }
        } catch (error) {
            console.error('Error logging in user:', error);
        }
    };

    useEffect(() => {
        if (!attemptedQuickLogin) {
            console.log("Attempting quick login...");
            const storedUserID = localStorage.getItem("last_userID");
            const storedPassword = localStorage.getItem("last_password");
            if (storedUserID && storedPassword) {
                handleUserLogin(storedUserID, storedPassword);
                console.log(`Quick login with userID: ${storedUserID}`);
            }
            setAttemptedQuickLogin(true);
        }

        console.log(`Connecting to ${backend.includes("local") ? "local" : backend} backend; with userID: ${userID}`);
    }, [backend, userID]);

    return (
        <AppContext.Provider value={{ backend, userID, handleUserLogin, addUser, admins }}>
            {children}
        </AppContext.Provider>
    );
};

export const useAppContext = (): AppContextType => {
    const context = useContext(AppContext);
    if (!context) {
        throw new Error('useAppContext must be used within an AppProvider');
    }
    return context;
};
