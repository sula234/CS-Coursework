.data
	strPow1: .asciiz "My power works, proof #"
	strLines: .asciiz "-----------------------\n"
	strPow2: .asciiz "When the n is "
	strPow3: .asciiz "With power as "
	strResult: .asciiz "The result is "
	
	factorial1: .float 1
.text

main:	
	
	# s3 contains num of iteration for loop
	li $s3 4
	# s0 = i 
	li $s0 1
	
	loop:
	
	# print first line
	la $a0 strPow1
	li $v0 4
	syscall
	
	move $a0 $s0
	li $v0 1
	syscall
	
	# print EOL
	addi $a0 $zero 10
	li $v0 11
	syscall
	
	la $a0 strPow2
	li $v0 4
	syscall
	
	# input float
	li $v0 6
	syscall
	mov.s $f12 $f0
	
	la $a0 strPow3
	li $v0 4
	syscall
	
	li $v0 5
	syscall
	move $a0 $v0
	
	# call procedure float Power (float f = f12, int n = a0)
	jal Power
	
	la $a0 strResult
	li $v0 4
	syscall
	
	# print float result from Factorial
	mov.s $f12 $f0
	li $v0 2
	syscall
	
	beq $s0 $s3 endProgram
	addi $s0 $s0 1
	
	# print \n
	addi $a0 $zero 10
	li $v0 11
	syscall
	
	# print lines
	la $a0 strLines
	li $v0 4
	syscall
	
	j loop
	
	endProgram:
	# end of program
	li $v0 10
	syscall
	
.globl Power	
# params: float a in $f12, int n in $a0
# return: float n in $f0 reg
Power:
	
	addi $sp, $sp, -20
	s.s $f20 0($sp)
	sw $a0 4($sp)
	s.s $f12 8($sp)
	sw $ra 12($sp)
	s.s $f22 16($sp)
	
	bne $a0 $zero repeate
	
	l.s $f20 0($sp)
	lw $a0 4($sp)
	l.s $f12 8($sp)
	lw $ra 12($sp)
	l.s $f22 16($sp)
	addi $sp, $sp, 20
	
	l.s $f0 factorial1
	jr $ra
	
repeate:
	slt $t1 $a0 $zero
	beq $t1 $zero n_positive
	
	# for negative power
	li $t1 -1
	mul $a0 $a0 $t1
	
	jal Power
	l.s $f20 factorial1
	div.s $f0 $f20 $f0
	
	l.s $f20 0($sp)
	lw $a0 4($sp)
	l.s $f12 8($sp)
	lw $ra 12($sp)
	l.s $f22 16($sp)
	addi $sp, $sp, 20
	
	jr $ra
	
n_positive:
	addi $a0 $a0 -1
	mov.s $f22 $f12
	
	jal Power
	mul.s $f0 $f0 $f22
	
	l.s $f20 0($sp)
	lw $a0 4($sp)
	l.s $f12 8($sp)
	lw $ra 12($sp)
	l.s $f22 16($sp)
	addi $sp, $sp, 20
	
	jr $ra
	 
	
	
	
	
	
