.data 
	input: .space 51
	inpSize: .word 50
	prompt: .asciiz "Please, enter a string: "
	output: .asciiz "The Uppercased version: "
	
.text	
	# print helper function
	li $v0, 4
	la $a0, prompt
	syscall
	# input the string
	li $v0, 8
	la $a0, input
	lw $a1, inpSize
	syscall	
loop:
  	# load byte from input to $s1
  	lb $s1, 0($a0)
  	# if (byte == \n): exit
  	beq $s1, 10, exit
  	# subtract from $s1 to get Uppercase
  	subi $s1, $s1, 32
  	# save Uppercased letter to data
  	sb $s1, 0($a0)
  	# increment loop by 1
  	addi $a0, $a0, 1
	# jump to loop
    	j loop

exit:
	# print helper message
	li $v0, 4
	la $a0, output
	syscall
	# print Uppercased string
	li $v0, 4
	la $a0, input
	syscall
	# exit program
	li $v0, 10
	syscall
