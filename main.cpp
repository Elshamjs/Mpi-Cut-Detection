#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <sstream>
#include <algorithm>
#include <string>


//<paralelo>
    int THIS_RANK;
    int NPROCS;
//</paralelo>

    std::string video_name = "";

class FrameDiffHistogram
{
public:
    int frame_number;
    long double distance_between_prev;


    FrameDiffHistogram() { frame_number = 0; distance_between_prev = 0; }

    FrameDiffHistogram(int frame_number, long double distance_between_prev)
    {
        this->distance_between_prev = distance_between_prev;
        this->frame_number = frame_number;
    }

    std::string toString()
    {
        return "{{ frame_number: " + std::to_string(frame_number) + " }{distance_between_prev: " + std::to_string(distance_between_prev) + " }}";
    }
};


long double erfinv(long double x) //La tabla z
{
    long double w, p;
    x = std::min(std::max(x, -0.99999L), 0.99999L);
    w = -std::log((1.0L - x) * (1.0L + x));
    if (w < 5.0L)
    {
        w = w - 2.5L;
        p = 2.81022636e-08L;
        p = 3.43273939e-07L + p * w;
        p = -3.5233877e-06L + p * w;
        p = -4.39150654e-06L + p * w;
        p = 0.00021858087L + p * w;
        p = -0.00125372503L + p * w;
        p = -0.00417768164L + p * w;
        p = 0.246640727L + p * w;
        p = 1.50140941L + p * w;
    }
    else
    {
        w = std::sqrt(w) - 3.0L;
        p = -0.000200214257L;
        p = 0.000100950558L + p * w;
        p = 0.00134934322L + p * w;
        p = -0.00367342844L + p * w;
        p = 0.00573950773L + p * w;
        p = -0.0076224613L + p * w;
        p = 0.00943887047L + p * w;
        p = 1.00167406L + p * w;
        p = 2.83297682L + p * w;
    }
    return p * x;
}


void saveFrameCuts(const std::vector<FrameDiffHistogram>& frame_cuts, std::string path_file) {
    printf("Guardando..\n");
    std::ofstream archivo(path_file, std::ios::binary);
    if (!archivo.is_open()) {
        std::cout << "Error al abrir el archivo para escritura: " << path_file << std::endl;
        return;
    }

    int num_frames = frame_cuts.size();
    archivo.write(reinterpret_cast<const char*>(&num_frames), sizeof(int));

    for (const auto& frame_cut : frame_cuts) {
        archivo.write(reinterpret_cast<const char*>(&frame_cut), sizeof(FrameDiffHistogram));
    }

    archivo.close();
    std::cout << "Archivo guardado correctamente: " << path_file << std::endl;
}

std::vector<FrameDiffHistogram> loadFrameCuts(std::string ruta_archivo) {
    printf("Cargando..\n");
    std::ifstream archivo(ruta_archivo, std::ios::binary);
    if (!archivo.is_open()) {
        std::cout << "Error al abrir el archivo para lectura: " << ruta_archivo << std::endl;
        return std::vector<FrameDiffHistogram>();
    }

    int num_frames;
    archivo.read(reinterpret_cast<char*>(&num_frames), sizeof(int));

    std::vector<FrameDiffHistogram> frame_cuts(num_frames);
    for (auto& frame_cut : frame_cuts) {
        archivo.read(reinterpret_cast<char*>(&frame_cut), sizeof(FrameDiffHistogram));
    }

    archivo.close();
    std::cout << "Archivo cargado correctamente: " << ruta_archivo << std::endl;

    return frame_cuts;
}
cv::Mat convertToHSVL(cv::Mat img_rgb)
{
    cv::Mat img_hsv;
    cv::cvtColor(img_rgb, img_hsv, cv::COLOR_BGR2HSV_FULL);

    std::vector<cv::Mat> hsv_channels;
    cv::split(img_hsv, hsv_channels);
    cv::Mat hue = hsv_channels[0];
    cv::Mat saturation = hsv_channels[1];
    cv::Mat value = hsv_channels[2];

    cv::Mat lightness = (value + saturation) / 2;

    cv::normalize(lightness, lightness, 0, 255, cv::NORM_MINMAX);
    cv::normalize(hue, hue, 0, 255, cv::NORM_MINMAX);
    cv::normalize(saturation, saturation, 0, 255, cv::NORM_MINMAX);
    cv::normalize(value, value, 0, 255, cv::NORM_MINMAX);
    hsv_channels.push_back(lightness);
    cv::Mat img_hsvl;
    cv::merge(hsv_channels, img_hsvl);

    hsv_channels.clear();
    hue.release();
    saturation.release();
    value.release();
    lightness.release();

    return img_hsvl;
}

std::vector<FrameDiffHistogram> operationGatherAllFrames(std::vector<FrameDiffHistogram> local_frame_diff) //La idea es juntar todos los arrays en el proceso 0
{
    int local_size = local_frame_diff.size(); //primero el tamaño de cada array local
    std::vector<int> arrays_size(NPROCS);  //aca se guardaran los tamaños de cada array local, es un parametro que necesita saber el rank 0 para hacer MPI_Gatherv
    std::vector<int> array_starts(NPROCS); //aca se guarda que parte del array final ocuparan cada array local.

    MPI_Datatype FRAME_CUT_HISTOGRAM_TYPE; //MPI debe saber que formato tiene la memoria reservada por la clase FrameCutHistogram.
    MPI_Type_contiguous(sizeof(FrameDiffHistogram), MPI_BYTE, &FRAME_CUT_HISTOGRAM_TYPE); //Basicamente hay que decirle que de donde a donde esta reservada los bloques continuos de memoria
    MPI_Type_commit(&FRAME_CUT_HISTOGRAM_TYPE);

    MPI_Allgather(&local_size, 1, MPI_INT, arrays_size.data(), 1, MPI_INT, MPI_COMM_WORLD); //Se llena el array arrays_size con los local_size y luego se reparte a todos los demas procesos
    array_starts[0] = 0;
    for (int i = 1; i < NPROCS; i++) {
        array_starts[i] = array_starts[i - 1] + arrays_size[i - 1];
    }
    int full_array_size = array_starts[NPROCS - 1] + arrays_size[NPROCS - 1];

    FrameDiffHistogram* full_frame_diff= nullptr;
    if(THIS_RANK==0) full_frame_diff = new FrameDiffHistogram[full_array_size]; //aca se metera todo

    //Este metodo entonces va ocupar: el puntero del array local de donde se sacara la data, el size de ese puntero local, el tipo de dato, el puntero donde se unira todo, el puntero donde esta los tamaños de cada array local, el puntero donde esta todos los inicios, el tipo de llegada, el root y el comm
    MPI_Gatherv(local_frame_diff.data(), local_size, FRAME_CUT_HISTOGRAM_TYPE, full_frame_diff, arrays_size.data(), array_starts.data(), FRAME_CUT_HISTOGRAM_TYPE, 0, MPI_COMM_WORLD);
    MPI_Type_free(&FRAME_CUT_HISTOGRAM_TYPE);

    std::vector<FrameDiffHistogram> ret_array;
    if(THIS_RANK==0) ret_array = std::vector<FrameDiffHistogram>(full_frame_diff, full_frame_diff + full_array_size); // Por ultimo el puntero del array, lo metemos en un std::vector para manejarlo mejor
    return ret_array;
}


void processVideo()
{
    if(THIS_RANK==0) printf("\n\nProcesando..\n\n");
    std::vector<FrameDiffHistogram> local_frame_diff;
    cv::VideoCapture capturer("resources/"+video_name+".mp4");
    if (THIS_RANK == 0)
    {
        if (!capturer.isOpened())
        {
            printf("\n\nNo se pudo abrir el archivo de video.\n\n");
            return;
        }
        else
        {
            printf("\n\nSe pudo abrir\n\n");
        }
    }

    //Se definen los parametros para calcular el histograma
    static int channels[] = { 0, 1, 2, 3 };
    static int histSize[] = { 100, 100, 100, 100 };
    static float hranges[] = { 0, 255 };
    static float sranges[] = { 0, 255 };
    static float vranges[] = { 0, 255 };
    static float lranges[] = { 0, 255 };
    static const float* ranges[] = { hranges, sranges, vranges, lranges };

    //El histograma anterior y el actual, se saca la diferencia entre ellos.
    cv::Mat* hist_old = new cv::Mat();
    cv::Mat* hist_new = new cv::Mat();

    //Variable que captura el frame
    cv::Mat frame;

    //Se guarda el progreso actual
    float current_progress = 0;

    //Es para calcular el nuevo progreso despues de cada interacion
    int progress_percentage=0;

    //Literal lo que dice, obtener los frames totales del video
    unsigned long total_frames = capturer.get(cv::CAP_PROP_FRAME_COUNT);

    //Aca se guarda el frame capturado pero en formato de color hsv
    cv::Mat hsvl_frame;

    //<parelelo>
    int local_curr_frame_number = 0; //El numero de frame que esta procesando el proceso
    int local_processed_frames = 0; //La cantidad total de frames que ya se han procesado, es para sacar el procentaje restante
    if (THIS_RANK == 0) std::cout << "FRAMES TOTALES: " << total_frames << std::endl;
    int res= total_frames % NPROCS; //Aca se guarda el residuo de dividir la cantidad de procesos entre la cantidad de frames totales. Para hacer calculos
    int local_frame_start, local_frame_end; //Los procesos solo procesan una parte del video, local_frame_start y local_frame_end son los rangos de donde ha donde lo van hacer.
    int last_end = (total_frames / NPROCS) + (0 < res); //Los siguientes calculos definiran cada pedazo que le toca a cada proceso, esta variable guarda el acumulado.
    if (THIS_RANK == 0)
    {

        for (int i = 1; i < NPROCS; i++) //Inicia desde 1 porque el proceso 0 ya se cuales son sus rangos, se sacan aparte abajito.
        {
            local_frame_start = last_end + 1; //En last_end se guarda el acumulado de donde termino el ultimo rango del proceso anterior, ej. si el rank 5 va 200 - 300, last_end es 300, luego para sacar el rank 6 le sumo a last_end un 1 y ya tengo el inicio del rank6 que es 301
            local_frame_end = last_end + (total_frames / NPROCS) + (i < res); //Esta formula es para sacar el rango final de cada proceso
            last_end = local_frame_end;
            if (i == NPROCS - 1) local_frame_end--; 
            MPI_Send(&local_frame_start, 1, MPI_INT, i, 0, MPI_COMM_WORLD); //Una vez que el proceso 0 calcula el rango para x proceso, esos datos se mandan a el proceso x en cuestion
            MPI_Send(&local_frame_end, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        local_frame_start = 0; //ACA, se sacan los rangos del proceso 0
        local_frame_end = (total_frames / NPROCS) + (0 < res);
        if (NPROCS == 1) local_frame_end--;
        
    }
    if (THIS_RANK != 0)
    {
        MPI_Recv(&local_frame_start, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //Aca es donde los demas procesos toman sus rangos que son enviados desde el proceso 0
        MPI_Recv(&local_frame_end, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    for (int i = 0; i < NPROCS; i++)
    {
        if (i == THIS_RANK)
        {
            std::cout << "\n\nRANK " << THIS_RANK << " | " << local_frame_start << " <-> " << local_frame_end << std::endl << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    //</paralelo>

    local_curr_frame_number = (local_frame_start == 0 ? 0 : local_frame_start - 1);  //Aca se establese donde inicia este proceso. Si el frame de inicio es el 0, se deja en 0, en caso contrario se le resta uno. Esto porque si el rango es 200 - 300, para medir la distancia de 200, debo medirlo con el 199, pero antes debo sacarle el histograma al 199.
    capturer.set(cv::CAP_PROP_POS_FRAMES, local_curr_frame_number); //Y inmediatamente se setea el puntero de la capturadora en ese frame especifico

    while (local_curr_frame_number <= local_frame_end)
    {
        capturer.read(frame); //se captura ese frame 
        hsvl_frame= convertToHSVL(frame); //se convierte el frame a otro formato de color
        cv::calcHist(&hsvl_frame, 1, channels, cv::Mat(), *hist_new, 4, histSize, ranges); //se calcula el histograma
        if (local_curr_frame_number!=0) //Como siempre se calcula la distancia entre el frame actual y el anterior, el 0 no tiene anterior, por lo cual no se calcula la distancia
        {
            if (local_curr_frame_number != local_frame_start - 1) //Si a un proceso le toca el rango 200 - 300, eso quiere decir que si quiero medir la distancia que tiene el frame 200, tendria que medirlo con el frame 199. Este if es para eso. Si estoy en el frame 199 no puedo medir su distancia, tan solo quiero su histograma
            {
                local_frame_diff.push_back(FrameDiffHistogram(local_curr_frame_number, cv::compareHist(*hist_old, *hist_new, cv::HISTCMP_BHATTACHARYYA))); //Guardo el numero de frame y la distancia
            }
        }
        else
        {
            local_frame_diff.push_back(FrameDiffHistogram(local_curr_frame_number, 0)); //Si es el frame 0 la distancia es 0
        }
        hist_old->release(); //libero memoria
        hist_old = hist_new; //ahora el new pasa a ser el old
        hist_new = new cv::Mat();
        local_curr_frame_number++; //Aumento los contadores
        local_processed_frames++;
        current_progress = ((float)local_processed_frames / (float)(local_frame_end-local_frame_start)) * 100; //Esto es para sacar el progreso actual
        if (THIS_RANK == 0 && (int)current_progress > progress_percentage && (int)current_progress<100)
        {
            progress_percentage = current_progress;
            std::cout << "Porcentaje " << progress_percentage << " %" << std::endl;
        }
    }
    capturer.release(); //Libero la capturadora de frames

    //<paralelo>
    if(THIS_RANK==0) std::cout << "Porcentaje 100%" << std::endl;
    std::vector<FrameDiffHistogram> all_frame_diff = operationGatherAllFrames(local_frame_diff); //Cada array de cada proceso con los frames procesados se van a unir en un solo array

    if (THIS_RANK == 0)
    {
        for (auto& data : all_frame_diff)
        {
            std::cout << data.toString() << std::endl; //se imprime el vector resultante
        } 
        saveFrameCuts(all_frame_diff, "resources/" + video_name + ".bin"); //Se guarda el vector en un .bin
    }
    //</paralelo>
}

long double calculateTolerance(const std::vector<FrameDiffHistogram>& vec, long double confidence_level)
{
    int n = vec.size(); 

    long double avg = 0.0;
    for (const FrameDiffHistogram& obj : vec)
    {
        avg += obj.distance_between_prev;
    }
    avg /= n; 
    long double sumaCuadradosDiferencias = 0.0;
    int count = 0;
    for (const FrameDiffHistogram& obj : vec)
    {
        if (obj.distance_between_prev > avg)
        {
            sumaCuadradosDiferencias += pow(obj.distance_between_prev - avg, 2);
            count++;
        }
    }
    long double varianza = sumaCuadradosDiferencias / count; 

    long double desviacionEstandar = sqrt(varianza);

    long double extremoMayor = avg + erfinv(confidence_level) * desviacionEstandar;

    return avg+extremoMayor; 
}

std::vector<FrameDiffHistogram> cutArray(const std::vector<FrameDiffHistogram>& vec, int cut_at, int cut_to)
{
    if (cut_at < 0 || cut_at >= vec.size() || cut_to < 0 || cut_to >= vec.size() || cut_at > cut_to) {
        return std::vector<FrameDiffHistogram>();
    }

    std::vector<FrameDiffHistogram> vec_cortado;

    for (int i = cut_at; i <= cut_to; i++) {
        vec_cortado.push_back(vec[i]);
    }

    return vec_cortado;
}

void getVideoCuts()
{
    std::vector<FrameDiffHistogram> frame_cuts;
    std::vector<FrameDiffHistogram> camera_cuts;
    std::vector<FrameDiffHistogram> frame_cuts_current;

    cv::VideoCapture capturer("resources/" + video_name + ".mp4");
    frame_cuts = loadFrameCuts("resources/" + video_name + ".bin");
    frame_cuts[0].distance_between_prev = frame_cuts[1].distance_between_prev;
    long double tolerance;
    int last_cut_pos = 0;
    for (int current_frame_pos = 10; current_frame_pos < frame_cuts.size(); current_frame_pos++)
    {
        tolerance = calculateTolerance(cutArray(frame_cuts, last_cut_pos, current_frame_pos), 0.98);
        if (frame_cuts[current_frame_pos].distance_between_prev > tolerance)
        {
            camera_cuts.push_back(frame_cuts[current_frame_pos]);
            last_cut_pos = current_frame_pos;
            current_frame_pos += 10;
        }
    }
    int cut_number = 1;
    std::string path;
    if (std::filesystem::exists("resources/" + video_name)) std::filesystem::remove_all("resources/" + video_name);
    std::filesystem::create_directory("resources/" + video_name);
    for (auto& camera_cut : camera_cuts)
    {
        if (camera_cut.frame_number != 0)
        {
            path = "resources/" + video_name + "/cut" + std::to_string(cut_number);
            std::filesystem::create_directory(path);
            capturer.set(cv::CAP_PROP_POS_FRAMES, camera_cut.frame_number - 1);
            cv::Mat frame;
            capturer.read(frame);
            cv::imwrite("resources/" + video_name + "/cut" + std::to_string(cut_number) + "/1prev_" + std::to_string(camera_cut.frame_number) + ".jpg", frame);
            capturer.read(frame);
            cv::imwrite("resources/" + video_name + "/cut" + std::to_string(cut_number) + "/2cut_" + std::to_string(camera_cut.frame_number) + "_" + std::to_string((double)camera_cut.distance_between_prev) + ".jpg", frame);
            cut_number++;
        }
    }
}


int main(int argc, char* argv[]) 
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &THIS_RANK);
    MPI_Comm_size(MPI_COMM_WORLD, &NPROCS);
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
    video_name = "video5";


    processVideo();
    //getVideoCuts();

    
    std::cout << "RANK " << THIS_RANK << "\n";
    MPI_Finalize();
}

