import express from 'express';
import cors from 'cors';
import path from 'path';
import { fileURLToPath } from 'url';

//importar las rutas
import audioRoutes from './routes/audioRoutes.js';
//definir los módulos de entrada
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);


//definir permisos
const corsOptions = {
    origin: '*', //direccion del dominio del servidor
    methods:['GET', 'POST', 'PUT', 'PATCH', 'DELETE'],
    Credentials: true
}


const app = express();
app.use(cors(corsOptions));
app.use(express.json()); //interpreta objetos json
app.use(express.urlencoded({extended: true})) //se añade para mostrar formularios
app.use('/uploads', express.static(path.join(__dirname, '../uploads')));
//indicar que rutas se utiliza
app.use('/api', audioRoutes)
app.use((req, resp, next)=>{
    resp.status(400).json({
        message: 'Endpoint not found'
    })
})

export default app;