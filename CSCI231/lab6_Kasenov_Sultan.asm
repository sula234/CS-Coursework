.data
	length: .word 7
	array: .space 28
	target: .word 231
	promt1: .asciiz "Please, enter "
	promt2: .asciiz " integers:\n"
	output1: .asciiz "["
	output2: .asciiz ","
	output3: .asciiz "]"
.text

main:
	# print promt message
	la $a0, promt1
	li $v0, 4
	syscall
	
	lw $a0, length
	li $v0, 1
	syscall
	
	la $a0, promt2
	li $v0, 4
	syscall
	
	# fill array with input values
	lw $s0, length
	la $s1, array
	loop:
	
	li $v0, 5
	syscall
	sw $v0, 0($s1)
	
	addi $s1, $s1, 4
	addi $t0, $t0, 1
	bne $t0, $s0, loop
	
	# call twoSum() function
	la $a0, array
	lw $a1, length
	lw $a2, target
	jal twoSum
	
	move $s0, $v0
	move $s1, $v1
	
	# print result 
	li $v0, 4
	la $a0, output1
	syscall
	
	li $v0, 1
	move $a0, $s0
	syscall
	
	li $v0, 4
	la $a0, output2
	syscall
	
	li $v0, 1
	move $a0, $s1
	syscall
	
	li $v0, 4
	la $a0, output3
	syscall
	
	li $v0, 10
	syscall
	
	# int* twoSum(int* a, int size , int target );
twoSum:
	# save to stack 
	addi $sp, $sp, -24
	sw $s0, 0($sp)
	sw $s1, 4($sp)
	sw $s2, 8($sp)
	sw $s3, 12($sp)
	sw $s4, 16($sp)
	sw $s5, 20($sp)
	
	# first loop for(i = 0; i != length - 1; i++)
	move $s3, $zero
	loop1:
	# s0 = a[i]
	lw $s0, 0($a0)
	addi $a0, $a0, 4
	
	# second loop for(j = (i + 1); j != length; j++)
	move $s2, $a0
	addi $s4, $s3, 1
	loop2:
	# s1 = a[j]
	lw $s1, 0($s2)
	
	# if(a[i] + a[j] == target) : break loops
	add $s5, $s0, $s1
	beq $s5, $a2, exit
	# else increment adress and j
	addi $s2, $s2, 4
	addi $s4, $s4, 1
	
	# j != length and this the last instruction in loop2
	bne $s4, $a1, loop2
	
	# i++ and jump to loop 1
	addi $s3, $s3, 1
	j loop1

exit:
	move $v0, $s3
	move $v1, $s4

	# load values from stack
	lw $s0, 0($sp)
	lw $s1, 4($sp)
	lw $s2, 8($sp)
	lw $s3, 12($sp)
	lw $s4, 16($sp)
	lw $s5, 20($sp)
	addi $sp, $sp, 24

	jr $ra

