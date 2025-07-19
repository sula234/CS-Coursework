.data

	strInvalid: .asciiz "Invalid input!"
	strFact1: .asciiz "My factorial works, proof #"
	strLines: .asciiz "---------------------------\n"
	strFact2: .asciiz "When the n is "
	strResult: .asciiz "The result is "
	
	factorial1: .float 1
	factorial0: .float 0
.text
	
main:	
	
	# s3 contains num of iteration for loop
	li $s3 4
	# s0 = i 
	li $s0 1
	
	loop:
	
	# print first line
	la $a0 strFact1
	li $v0 4
	syscall
	
	move $a0 $s0
	li $v0 1
	syscall
	
	# print EOL
	addi $a0 $zero 10
	li $v0 11
	syscall
	
	la $a0 strFact2
	li $v0 4
	syscall
	
	# input float
	li $v0 6
	syscall
	
	la $a0 strResult
	li $v0 4
	syscall
	
	mov.s $f12 $f0
	
	# call procedure float Factorial(float n = f12)
	jal Factorial
	
	c.eq.s $f0 $f29
	bc1t catch
	
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
	
catch:
	beq $s0 $s3 endProgram
	addi $s0 $s0 1
	
	j loop
	
	endProgram:
	# end of program
	li $v0 10
	syscall

# params: float n in $f12
# return: float n in $f0 reg
.globl Factorial
Factorial:

	addi $sp, $sp, -20
	s.s $f20 0($sp)
	s.s $f22 4($sp)
	s.s $f12 8($sp)
	sw $ra 12($sp)
	s.s $f27 16($sp)
	
	l.s $f27 factorial0
	c.lt.s $f12 $f27
	bc1t Except
	
	# if (n > 1) 
	l.s $f20 factorial1
	c.lt.s $f20 $f12
	bc1t repeate 
	
	l.s $f20 0($sp)
	l.s $f22 4($sp)
	l.s $f12 8($sp)
	lw $ra 12($sp)
	l.s $f27 16($sp)
	addi $sp, $sp, 20
	
	l.s $f0 factorial1
	jr $ra
	
repeate:
	mov.s $f22 $f12
	sub.s $f12 $f12 $f20
	
	jal Factorial
	mul.s $f0 $f0 $f22
	
	l.s $f20 0($sp)
	l.s $f22 4($sp)
	l.s $f12 8($sp)
	lw $ra 12($sp)
	l.s $f27 16($sp)
	addi $sp, $sp, 20
	
	jr $ra


Except:
	la $a0 strInvalid
	li $v0 4
	syscall
	
	# print \n
	addi $a0 $zero 10
	li $v0 11
	syscall
	
	l.s $f0 factorial0
	
	jr $ra

