#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_COMMANDS 15
#define MAX_COMMAND_LENGTH 50
#define MAX_ARGS 10
#define MAX_ARG_LENGTH 40


/* struct to define command string and args to pass to the process*/
typedef struct {
    char command[MAX_ARG_LENGTH];
    char *args[MAX_ARGS];
} ParsedCommand;

/* This function was implemented to parse a command line with arguments string and, as a result, obtain a command 
line and an array of arguments in order to send it to the syscall function */
void parse_command(const char *input, ParsedCommand *parsedCommand) {

    // helper variables
    char *token;
    char commandCopy[MAX_ARG_LENGTH];
    size_t argCount = 1;

    // since example contains such strange requirements
    parsedCommand->args[0] = "command";

    // Copy the input command to preserve it
    strncpy(commandCopy, input, sizeof(commandCopy) - 1);
    commandCopy[sizeof(commandCopy) - 1] = '\0';

    // Extract the command
    token = strtok(commandCopy, " ");
    if (token != NULL) {
        strncpy(parsedCommand->command, token, sizeof(parsedCommand->command) - 1);
        parsedCommand->command[sizeof(parsedCommand->command) - 1] = '\0';

        // Extract arguments
        token = strtok(NULL, " ");
        while (token != NULL && argCount < MAX_ARGS) {
            parsedCommand->args[argCount] = strdup(token);
            argCount++;
            token = strtok(NULL, " ");
        }
    }

    // Add a NULL terminator to the args array (also requierements)
    parsedCommand->args[argCount] = NULL;
}

/*delete first space from command because after the 
    first parsing the command is separated by commas, 
    the user can write the command starting with a space*/
void check_for_space(char commands[][MAX_COMMAND_LENGTH], size_t commands_num) {
        for (size_t i = 0; i < commands_num; i++) {

            if(commands[i][0] == ' ') {
                for (int k = 0; k < MAX_COMMAND_LENGTH - 1; k++) 
                    commands[i][k] = commands[i][k + 1];
            }
    }
}

/* define path to executable file located in bin*/
void mergeWithBin(char commands[][MAX_COMMAND_LENGTH], size_t commands_num) {
    for (size_t i = 0; i < commands_num; i++) {
        // Create a temporary buffer to hold the result
        char tmp[MAX_COMMAND_LENGTH + 5];  // 5 for "/bin/"

        // Copy "/bin/" to the temporary buffer
        strcpy(tmp, "/bin/");

        // Concatenate the command to the temporary buffer
        strcat(tmp, commands[i]);

        // Copy the result back to the original array
        strncpy(commands[i], tmp, MAX_COMMAND_LENGTH);
    }
}

/* this function creates child processes from the main process and replaces the code 
    with commands specified by the user*/
void executeCommands(char commands[][MAX_COMMAND_LENGTH], size_t commands_num) {

    ParsedCommand parsedCommand;

	for(int i=0; i<commands_num; i++) {

        if(fork() == 0) {
            parse_command(commands[i], &parsedCommand);
			execvp(parsedCommand.command, parsedCommand.args);
            exit(0);
        }
    }
    for(int i=0; i<commands_num; i++) 
    	wait(NULL);
    
}


int main() {
    char commands[MAX_COMMANDS][MAX_COMMAND_LENGTH];
    char input[MAX_COMMAND_LENGTH];
    char arguments[MAX_COMMANDS][MAX_ARGS][MAX_ARG_LENGTH];
    int commands_num = 0;

    while(1) {
        printf("\nEnter commands separated by commas (to exit type 'exit()' ): ");
        fgets(input, sizeof(input), stdin);

        // Remove trailing newline character
        input[strcspn(input, "\n")] = '\0';

        if (strcmp(input, "exit()") == 0) {
            printf("Exiting the main process...\n");
            break;  // Exit the loop if the input is "exit"
        }

        // Tokenize the input commands
        char *token = strtok(input, ",");
        while (token != NULL) {

            // Store each command in the array
            strncpy(commands[commands_num], token, sizeof(commands[commands_num]) - 1);
            commands[commands_num][sizeof(commands[commands_num]) - 1] = '\0';
            commands_num++;

            token = strtok(NULL, ",");
        }

        check_for_space(commands, commands_num);

        // Call the function to merge with "/bin/"
        mergeWithBin(commands, commands_num);

        // Execute each command
        executeCommands(commands, commands_num);
    }

    return 0;
}
