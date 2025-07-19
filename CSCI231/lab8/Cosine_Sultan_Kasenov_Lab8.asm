.data

	strCosine1: .asciiz "Enter a degree: "
	strCosine2: .asciiz "A negative degree detected!"
	
	factorial1: .float -1
	factorial2: .float 2
	factorial3: .float 1
	factorial4: .float 0
	pi: .float 3.1415926535
	pi_1: .float 180
.text

main:

loop1:
	la $a0 strCosine1
	li $v0 4
	syscall
	
	li $v0 6
	syscall
	
	# check for negative number 
	l.s $f16 factorial4
	c.lt.s $f0 $f16
	bc1t exit1
	
	# call procedure
	mov.s $f12 $f0
	jal cosine
	
	# print result
	mov.s $f12 $f0
	li $v0 2
	syscall
	
	# print EOL
	addi $a0 $zero 10
	li $v0 11
	syscall
	
	j loop1
	
exit1:
	la $a0 strCosine2
	li $v0 4
	syscall
	
	# end program
	li $v0 10
	syscall
	
cosine:
	addi $sp, $sp, -28
	sw $s0 0($sp)
	sw $s1 4($sp)
	s.s $f21 8($sp) # for sum
	s.s $f22 12($sp) # for tmp sum
	s.s $f20 16($sp) # for x
	s.s $f25 20($sp) # for i in our loop
	sw $ra 24($sp) 
	
	# calc rads
	l.s $f16 pi
	l.s $f17 pi_1
	div.s $f18 $f16 $f17
	mul.s $f12 $f12 $f18
	
	li $s1 20
	
	# save x value
	mov.s $f20 $f12
	
	move $s0 $zero
	
loop2:
	beq $s0 $s1 exit2
	
	# calc (-1)^n
	l.s $f12 factorial1
	move $a0 $s0
	jal Power
	
	# save (-1)^n
	mov.s $f22 $f0
	
	# x^2n
	mov.s $f12 $f20
	li $t0 2
	mul $a0 $t0 $s0
	jal Power
	
	# save x^2n
	mul.s $f22 $f22 $f0
	
	# (2n)!
	mov.s $f12 $f25
	l.s $f16 factorial2
	mul.s $f12 $f12 $f16
	jal Factorial
	
	# save (2n)!
	div.s $f22 $f22 $f0
	
	# calc sum
	add.s $f21 $f21 $f22
	
	# Increment our loop by one. We also increment $f25 since we 
	# should pass n to Factorial procedure and params in Factorial have float type.
	addi $s0 $s0 1
	l.s $f18 factorial3
	add.s $f25 $f25 $f18
	j loop2
exit2:
	mov.s $f0 $f21
	
	lw $s0 0($sp)
	lw $s1 4($sp)
	l.s $f21 8($sp)
	l.s $f22 12($sp)
	l.s $f20 16($sp)
	l.s $f25 20($sp)
	lw $ra 24($sp) 
	addi $sp, $sp, 28
	
	jr $ra
	
