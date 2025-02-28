import React, { createContext, useContext, useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';

interface AppContextType {
    backend: string;
    userID: string | null;
    handleUserLogin: (enteredUserID: string, password: string) => void;
    addUser: (userID: string, password: string) => void;
    admins: string[];
}

const AppContext = createContext<AppContextType | undefined>(undefined);

export const AppProvider: React.FC<{ backend: string; children: React.ReactNode }> = ({ backend, children }) => {
    const [userID, setUserID] = useState<string | null>("default");
    const [attemptedQuickLogin, setAttemptedQuickLogin] = useState(false);
    const [searchParams] = useSearchParams();
    const userParam = searchParams.get('user');
    const canvasParam = searchParams.get('canvas');

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
        if (userParam && !attemptedQuickLogin) {
            console.log("Attempting login from url...");
            handleUserLogin(userParam);
        } else if (!attemptedQuickLogin) {
            console.log("Attempting quick login from browser...");
            const storedUserID = localStorage.getItem("last_userID");
            const storedPassword = localStorage.getItem("last_password");
            if (storedUserID && storedPassword !== null) {
                handleUserLogin(storedUserID, storedPassword);
                console.log(`Quick login with userID: ${storedUserID}`);
            } else {
                console.log(`FAILED: Stored UserID: ${storedUserID}, Stored Password: ${storedPassword}`);
            }
        }
        setAttemptedQuickLogin(true);

        console.log(`Connecting to ${backend.includes("local") ? "local" : backend} backend; with userID: ${userID}`);
    }, [backend, userID, userParam, canvasParam]);

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
