.data
	arr1: .word 22 33 44 55 66 77
	len1: .word 6
	target: .word 35
	X: .asciiz "X = "
	Y: .asciiz "\nY = "
.text 

main:
	la $a0, arr1 # call function
	lw $a1, len1
	lw $a2, target
	jal countingBigSmall
	
	move $s0, $v0
	
	la $a0, X
	li $v0, 4
	syscall
	
	move $a0, $s0
	li $v0, 1
	syscall
	
	la $a0, Y
	li $v0, 4
	syscall
	
	move $a0, $v1
	li $v0, 1
	syscall
	
	li $v0, 10
	syscall
	
countingBigSmall: 
	move $v0, $zero # X
	move $v1, $zero # Y
	li $t1, -1
	
	move $t3, $a0
   loop:
   	sub $t4, $t3, $a0 # find i
   	srl $t4, $t4, 2
   	beq $t4, $a1, exit # if i == len : exit
   
   	lw $t0, 0($t3) # load element
   	addi $t3, $t3, 4 # add adress 4 bytes

   
	slt $t1, $a2, $t0 # is N < arr[i] 
	
	bne $t1, $zero, addY
	addi $v0, $v0, 1  # add 1 to X
	j loop
	
	addY:
	addi $v1, $v1, 1 # add 1 to Y
	j loop
	
	exit:
	jr $ra 
	
