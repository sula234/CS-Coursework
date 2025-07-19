#include <pthread.h>
#include <stdio.h> 
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include <unistd.h>


#define MAX_FILE_NUM 30
#define MAX_FILE_NAME_LENGTH 20
#define NUM_THREADS 3
#define FILES_PATH "my_files/"
#define SEARCH_STRING1 "CSCI332"
#define SEARCH_STRING2 "OS"

/*just helper struct to make code more clear*/
struct args {
    char* filename;
    pthread_t* workers;
};

/* threads call this function */
void *runner(void *params); 

/* traverse dir to get all files*/
void get_file_list(char file_list[MAX_FILE_NUM][MAX_FILE_NAME_LENGTH], size_t *file_num);

/* just merge with parent dir (my_files/) */
void merge_with_path(char file_list[][MAX_FILE_NAME_LENGTH]) {
    for (size_t i = 0; i < 10; i++) {
        // Create a temporary buffer to hold the result
        char tmp[MAX_FILE_NAME_LENGTH + 5];  // 5 for "/bin/"

        // Copy "/bin/" to the temporary buffer
        strcpy(tmp, FILES_PATH);

        // Concatenate the command to the temporary buffer
        strcat(tmp, file_list[i]);

        // Copy the result back to the original array
        strncpy(file_list[i], tmp, MAX_FILE_NAME_LENGTH);
    }
}


void get_file_list(char file_list[MAX_FILE_NUM][MAX_FILE_NAME_LENGTH], size_t *file_num) {
    DIR *d;
    struct dirent *dir;
    int i = 0;

    

    d = opendir(FILES_PATH);
    if (d)
    {
        while ((dir = readdir(d)) != NULL)
        {   
            if(i == 10) {
            strcpy(file_list[0], dir->d_name);
            }
            else if(i == 11) {
            strcpy(file_list[1], dir->d_name);
            }
             else {
             strcpy(file_list[i], dir->d_name);
             }
            i++;

        }
        closedir(d);
    }

    *file_num = i - 2;
}


int main(int argc, char *argv[]) {

    /* an array of threads to be joined upon */ 
    pthread_t workers[NUM_THREADS];

    pthread_attr_t attr; /* set of thread attributes */

    char file_list[MAX_FILE_NUM][MAX_FILE_NAME_LENGTH];
    size_t file_num = 0;

    get_file_list(file_list, &file_num);
    merge_with_path(file_list);

    for (size_t i = 0; i < file_num; i++) {
        printf("%s\n", file_list[i]);

        
    }
    /* get physical cores number */
    long number_of_processors = sysconf(_SC_NPROCESSORS_ONLN);
    for (int i = 0; i < number_of_processors; i++) {
        /* set the default attributes of the thread */ 
        pthread_attr_init(&attr);

        struct args *params = (struct args *)malloc(sizeof(struct args));
        params->filename = file_list[i];
        params->workers = workers;

        /* create the thread */
        pthread_create(&workers[i], &attr, runner, (void *)params);

    }

    for (int i = 0; i < NUM_THREADS; i++) 
        pthread_join(workers[i], NULL);

    printf("Termination of the main process\n");

}


/* The thread will execute in this function */ 
void *runner(void *params) {

    char line[1024];
    int line_number = 0;
    FILE *file;

    // Open the file
    file = fopen(((struct args*)params)->filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

        // Search for the string in the file
    while (fgets(line, sizeof(line), file) != NULL) {
        line_number++;

        if (strstr(line, SEARCH_STRING1) != NULL) {
            printf("String1 '%s' found in file %s at line %d\n", SEARCH_STRING1, ((struct args*)params)->filename, line_number);

            // Cancel other threads
                for (int i = 0; i < NUM_THREADS; i++) {
                    if (pthread_self() != ((struct args*)params)->workers[i]) {
                        pthread_cancel(((struct args*)params)->workers[i]);
                    }
                }
            printf("here");
            fclose(file);
            pthread_exit(0); 
        }

        if (strstr(line, SEARCH_STRING2) != NULL) {
            printf("String2 '%s' found in file %s at line %d\n", SEARCH_STRING2, ((struct args*)params)->filename, line_number);
        }
    }

    fclose(file);
    pthread_exit(0); 

}
