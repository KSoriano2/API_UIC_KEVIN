import { Router } from "express";
import multer from "multer";
import { clasificarAudio } from "../controladores/audioControlador.js";

// Configuración de multer para subir audios
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, "uploads/");
    },
    filename: (req, file, cb) => {
        const uniqueName = Date.now() + "-" + file.originalname;
        cb(null, uniqueName);
    }
});

const router = Router();

const upload = multer({ storage });

router.post("/clasificar", upload.single("audio"), clasificarAudio);

export default router;
