.data
	arr1: .word 1 2 3 4 5
	emptySpace1: .space 12
	arr2: .word 25, 35, 45, 55, 65, 75
	emptySpace2: .space 40
	len1: .word 5
	len2: .word 6
	
.text

main:	
	
	la $a0, arr1
	lw $a1, len1
	
	jal reverse
	
	la $a0, arr2
	lw $a1, len2
	
	jal reverse

	li $v0, 10
	syscall
	
reverse: # ($a0 = int* p, $a1 = size)
	li $t0, 0 # i = 0
	move $t3, $a0
	
	srl $t4, $a1, 1 #$t4 = size/2
	
	li $t5, 4
	mul $a1, $a1, $t5 #size in bytes
	mul $t4, $t4, $t5
	
	loop: # for( i = 0; i != size/2; i++)
	beq $t0, $t4, exit
	
	lw $s0, 0($t3) # $s0 = p[i]
	
	move  $t2, $a0
	add $t2 , $t2, $a1
	subi  $t2, $t2, 4
	sub $t2, $t2, $t0 # $t2 = &p[size - 1 - i]
	
	lw $s1, 0($t2) # $s1 = p[size - 1 - i]
	
	sw $s1, 0($t3) # p[i] = p[size - 1 - i]
	sw $s0, 0($t2) # p[size - 1 -i] = tmp
	
	addi $t0, $t0, 4 # i += 4
	add $t3, $a0, $t0 # increment adress by 1 word
	
	j loop
	
	exit:
	jr $ra 
	
	

	
	
	
	
	
	
	
	
	
	
	
	

