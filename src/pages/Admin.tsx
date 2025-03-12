import React, { useState, useEffect } from 'react';
import { useAppContext } from '../context/AppContext';

// const PasswordEntry: React.FC<{ onSubmit: (password: string) => void }> = ({ onSubmit }) => {
//     const [password, setPassword] = useState('');

//     const handlePasswordChange = (e: React.ChangeEvent<HTMLInputElement>) => {
//         setPassword(e.target.value);
//     };

//     const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
//         e.preventDefault();
//         onSubmit(password);
//     };

//     return (
//         <form onSubmit={handleSubmit}>
//             <label>
//                 Password:
//                 <input type="password" value={password} onChange={handlePasswordChange} />
//             </label>
//             <button type="submit">Submit</button>
//         </form>
//     );
// };

const Admin: React.FC = () => {
    const { backend, addUser } = useAppContext();
    //const [isAuthenticated, setIsAuthenticated] = useState(true); //TODO, turn the password back on by setting this to false 
    const [data, setData] = useState<any[]>([]);
    const [users, setUsers] = useState<any[]>([]);
    const [canvases, setCanvases] = useState<any[]>([]);
    const [selectedUser, setSelectedUser] = useState<string>('');
    const [selectedCanvas, setSelectedCanvas] = useState<string>('');
    const [canvasId, setCanvasId] = useState<string>('');
    const [timestamp, setTimestamp] = useState<string>('');
    const [rawData, setRawData] = useState<string>('');
    const [enteredUserID, setEnteredUserID] = useState<string>('');
    const [enteredPassword, setEnteredPassword] = useState<string>('');
    const [isAddUserOpen, setIsAddUserOpen] = useState<boolean>(false);

    // const handlePasswordSubmit = (password: string) => {
    //     if (password === 'miku') {
    //         setIsAuthenticated(true);
    //     } else {
    //         alert('Incorrect password');
    //     }
    // };
    const handleUserChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
        const userId = e.target.value;
        setSelectedUser(userId);
        setSelectedCanvas('');
        setCanvasId('');
        setTimestamp('');
        setRawData('');
        if (userId === 'browserdata') {
            const browserData = Object.keys(localStorage).filter(key => localStorage.getItem(key)?.includes('"nodes":'));
            setCanvases(browserData.map(key => ({ id: key, name: key })));
        } else {
            const response = await fetch(`${backend}/api/list-users`);
            const result = await response.json();
            if (result.success) {
                const user = result.users.find((user: any) => user.id === userId);
                setCanvases(user ? user.canvases : []);
            } else {
                console.error("Failed to fetch users:", result.error);
            }
        }
    };

    const isValidImageUrl = (url: string) => {
        return /\.(jpeg|jpg|gif|png|webp)$/.test(url);
    };

    const handleCanvasChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
        const canvasId = e.target.value;
        setSelectedCanvas(canvasId);
        if (selectedUser === 'browserdata') {
            const canvasData = JSON.parse(localStorage.getItem(canvasId) || '{}');
            setCanvasId(canvasId);
            setTimestamp(canvasData.timestamp || '');
            setRawData(JSON.stringify(canvasData, null, 2));
            setData(canvasData.nodes.map((node: any) => ({
                node_id: node.id,
                type: node.type,
                content: isValidImageUrl(node.data.content) ? <img src={node.data.content} alt="node content" style={{ maxWidth: '50px' }} /> : node.data.content.substring(0, 100)
            })));
        } else {
            const response = await fetch(`${backend}/api/get-canvas/${canvasId}`);
            const result = await response.json();
            if (result.success) {
                setCanvasId(result.canvas.id);
                setTimestamp(result.canvas.timestamp);
                setRawData(JSON.stringify(result.canvas, null, 2));
                setData(result.canvas.nodes.map((node: any) => ({
                    node_id: node.id,
                    type: node.type,
                    content: isValidImageUrl(node.data.content) ? <img src={node.data.content} alt="node content" style={{ maxWidth: '50px' }} /> : node.data.content.substring(0, 100)
                })));
            }
        }
    };

    const handleDeleteCanvas = async () => {
        if (selectedUser && selectedCanvas) {
            const isConfirmed = window.confirm('Are you sure you want to delete this canvas?');
            if (isConfirmed) {
                const response = await fetch(`${backend}/api/delete-canvas/${selectedUser}/${selectedCanvas}`, {
                    method: 'DELETE'
                });
                const result = await response.json();
                if (result.success) {
                    alert('Canvas deleted successfully');
                    setSelectedCanvas('');
                    setCanvasId('');
                    setTimestamp('');
                    setData([]);
                    setRawData('');
                } else {
                    alert('Failed to delete canvas');
                }
            }
        }
    };

    useEffect(() => {
        const fetchUsers = async () => {
            try {
                const response = await fetch(`${backend}/api/list-users`);
                const result = await response.json();
                if (result.success) {
                    setUsers(result.users);
                } else {
                    console.error("Failed to fetch users:", result.error);
                }
            } catch (error) {
                console.error("Error fetching users:", error);
            }
        };
        fetchUsers();
    }, []);

    return (
        <div className='flex flex-col mt-20 overflow-scroll px-10' style={{ height: 'calc(100vh - 100px)' }}>
            {/* {isAuthenticated ? ( */}
                <>
                    <div className='flex flex-wrap bg-stone-200 p-4'>
                        <div className='mr-4 mb-4'>
                            <h3 onClick={() => setIsAddUserOpen(!isAddUserOpen)} className="cursor-pointer">
                                {isAddUserOpen ? '▼' : '▶'} Add new user
                            </h3>
                            {isAddUserOpen && (
                                <div>
                                    <input
                                        type="text"
                                        placeholder="Enter new user ID"
                                        value={enteredUserID}
                                        onChange={(e) => setEnteredUserID(e.target.value)}
                                        className="w-full p-2 border rounded text-black mb-2"
                                    />
                                    <input
                                        type="password"
                                        placeholder="Enter new user password"
                                        value={enteredPassword}
                                        onChange={(e) => setEnteredPassword(e.target.value)}
                                        className="w-full p-2 border rounded text-black mb-2"
                                    />
                                    <button 
                                        onClick={() => addUser(enteredUserID, enteredPassword)} 
                                        className="w-full bg-stone-500 text-white p-2 rounded"
                                    >
                                        Add User
                                    </button>
                                </div>
                            )}
                        </div>
                        
                        <div className='mr-4 mb-4 display-block w-full p-3 relative'>
                        <button 
                                        onClick={() => console.log(localStorage)} 
                                        className="w-50% bg-brown-500 text-white p-2 rounded mt-2"
                                    >
                                        Print Local Storage to console
                                    </button>
                            <br></br>
                            <label htmlFor='user-select'>Select User: </label>
                            <select id='user-select' value={selectedUser} onChange={handleUserChange} className="w-full p-2 border rounded text-black mb-2">
                                <option value=''>Select a user</option>
                                {users.map(user => (
                                    <option key={user.id} value={user.id}>{user.id}</option>
                                ))}
                                <option value='browserdata'>Browser Data</option>
                            </select>
                        </div>
                        {selectedUser && (
                            <div className='mr-4 mb-4'>
                                <label htmlFor='canvas-select'>Select Canvas: </label>
                                <select id='canvas-select' value={selectedCanvas} onChange={handleCanvasChange} className="w-full p-2 border rounded text-black mb-2">
                                    <option value=''>Select a canvas</option>
                                    {canvases.map(canvas => (
                                        <option key={canvas.canvasId} value={canvas.canvasId}>{canvas.canvasName}</option>
                                    ))}
                                </select>
                            </div>
                        )}

                    </div>
                    {selectedCanvas && (
                        <div className='bg-brown-200 p-4'>
                            <div className='mb-4'>

                                <button 
                                    onClick={() => {

                                        console.log("sending user to: " + selectedUser + " and canvas to:" + canvasId);
                                        window.location.href = `/?user=${selectedUser}&canvas=${canvasId}`;
                                    }} 
                                    className='bg-blue-500 text-white px-4 py-2 mb-4'
                                >
                                    Go to Canvas Page
                                </button>

                                <p>Canvas ID in database: {canvasId}</p>

                                <p>Timestamp: {timestamp}</p>
                                <p>Number of Nodes: {data.length}</p>
                                <p className="font-bold">Raw canvas data dump from server: </p>
                                <div className='overflow-scroll bg-gray-100 p-4 mt-4 text-xs' style={{ height: '200px', width: '100%', whiteSpace: 'pre-wrap' }}>
                                    <pre>{rawData}</pre>
                                </div>

                                <button onClick={handleDeleteCanvas} className='bg-red-500 text-white px-4 py-2'>Delete Canvas</button>
                            </div>
                            <div className='overflow-scroll' style={{ height: 'calc(100vh - 100px)' }}>
                                <p className="font-bold">Node data from server, as a table: </p>
                                <table className='min-w-full'>
                                    <thead>
                                        <tr>
                                            <th className='px-4 py-2'></th>
                                            {data.length > 0 && Object.keys(data[0]).map((key) => (
                                                <th key={key} className='px-4 py-2'>{key}</th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {data.map((row, index) => (
                                            <tr key={index}>
                                                <td className='border px-4 py-2'>{index + 1}</td>
                                                {Object.values(row).map((value, idx) => (
                                                    <td key={idx} className='border px-4 py-2'>{value as React.ReactNode}</td>
                                                ))}
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}
                </>
            
            {/* // ) : (
            //    // <PasswordEntry onSubmit={handlePasswordSubmit} />
            // )} */}
        </div>
    );
};
export default Admin;