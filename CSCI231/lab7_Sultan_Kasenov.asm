.data
	promt: .asciiz "Give me a number: "
	output: .asciiz "Is it palindrome: "
	yes_message: .asciiz "YES"
	no_message: .asciiz "NO"
	
.text
	
main:
	# print promt message
	li $v0, 4
	la $a0, promt
	syscall
	
	# get a number 
	li $v0, 5
	syscall
	
	# move input number to s1
	move $s1, $v0
	
	# move int to $a0 
	add $a0, $v0, $zero
	
	# move 0 to a1
	move $a1, $zero
	
	# call function 
	jal Palindrome
	
	# save return to $s0
	move $s0, $v0
	
	# print output message
	li $v0, 4
	la $a0, output
	syscall
	
	# if return of function == input print YES otherwise NO
	bne $s0, $s1, NO
	
	la $a0 yes_message
	syscall
	
	# end program
	li $v0, 10
	syscall
	
NO:	la $a0, no_message
	syscall
	
	# end program
	li $v0, 10
	syscall
	
# params: input number(n) at $a0, temporary val(tmp) at $a1
# return reverse int number 
Palindrome:

	# save values to stack
	addi $sp, $sp, -16
	sw $s0, 0($sp)
	sw $s1, 4($sp)
	sw $ra, 8($sp)
	sw $s2, 12($sp)
	
	# if n == 0 exit
	beq $a0, $zero, exit
	
	# tmp * 10
	mul $s2, $a1, 10
	li $s0, 10
	
	div $a0, $s0
	# move remainder of n % 10 to $s1
	mfhi $s1
	
	# tmp = (tmp * 10) + (n % 10);
	add $a1, $s2, $s1
	
	div $a0, $a0, $s0
	jal Palindrome
	
	
exit:	
	# return values from stack
	lw $s0, 0($sp)
	lw $s1, 4($sp)
	lw $ra, 8($sp)
	lw $s2, 12($sp)
	addi $sp, $sp, 16

	# save tmp to $v0
	move $v0, $a1
	
	jr $ra
	
	
	
	
	
	
	
	
	
	
	
	
