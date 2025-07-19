.data
	#arr1: .space 64		# 4 by 4 - 64
	arr1: .space 1024		# 16 by 16 - 1024
	#arr1: .space 4096		# 32 by 32 - 4096
	#size: .word 64
	size: .word 1024
	#size: .word 4096
	rows: .word 16
	cols: .word 16

	arr2: .space 1024

	arr3: .space 1024
	mStr1: .asciiz "Matrix 1 is:\n"
	mStr2: .asciiz "\n\nMatrix 2 is:\n"
	mStr3: .asciiz "\n\nMatrix 3 is:\n"
.text


	la $a2 arr1
	addi $a1 $zero 20	# upper bound for random
	lw $t8 size
	
	add $t4 $a2 $t8
	jal randomizer		# randoming for matrix 1
	
	la $a2 arr2
	add $t4 $a2 $t8
	jal randomizer		# randoming for matrix 2
	
	la $a0 mStr1
	jal printString
	
	la $a0 arr1
	lw $a1 rows
	lw $a2 cols
	
	#addi $t1 $t1 2		# i
	#addi $t2 $t2 2		# j
	
	#sll $s0 $t1 2		# i * 4 (4 is the size of rows)
	#add $s0 $s0 $t2		# (i * 4) + j
	#sll $s0 $s0 2		# * 4, coz 4 bytes, then we get byte offset of [i][j]

	jal matrixPrinter		# printing 1st matrix
	
	la $a0 mStr2
	jal printString
	la $a0 arr2
	jal matrixPrinter		# printing 2nd matrix
	
	la $a0 arr1
	la $a1 arr2
	la $a2 arr3
	lw $a3 rows
	jal multiplyMatricesPlease	# need to implement
	
	la $a0 mStr3
	jal printString
	
	la $a0 arr3
	lw $a1 rows
	lw $a2 cols
	jal matrixPrinter			# printing 3rd matrix
	li $v0 10
	syscall
	
	multiplyMatricesPlease:	# 
		addi $sp $sp -32
		sw $s0 ($sp)
		sw $s1 4($sp)
		sw $s2 8($sp)
		sw $s3 12($sp)
		sw $s4 16($sp)
		sw $s5 20($sp)
		sw $s6 24($sp)
		sw $s7 28($sp)
		
		li $s0 -1 # i index
		li $s7 4 # contains size of 1 word or float(in bytes)
		loop1:
		li $s1 -1 # j index
		
		addi $s0 $s0 1
		beq $s0 $a3 exit_mult
			loop2:
				li $s2 -1 # k index
				
				add $s1 $s1 1
				beq $s1 $a3 loop1
				loop3:
					add $s2 $s2 1
					beq $s2 $a3 exit_loop3
					
					mul $s3 $a3 $s0 # start of calc adress for arr3[i][j]
					add $s3 $s3 $s1
					mul $s3 $s3 $s7
					add $s3 $s3 $a2 # s3 contains arr3[i][j] adress
					
					mul $s4 $a3 $s0 
					add $s4 $s4 $s2
					mul $s4 $s4 $s7
					add $s4 $s4 $a0 # s4 contains arr1[i][k] adress
					
					mul $s5 $a3 $s2 
					add $s5 $s5 $s1
					mul $s5 $s5 $s7
					add $s5 $s5 $a1 # s5 contains arr2[k][j] adress
					
					lw $t0 0($s4)
					lw $t1 0($s5)
					mul $t2 $t0 $t1 # find arr1[i][k] * arr2 [k][j] and save in $t2
					
					add $s6 $s6 $t2 # add result in $t2
					
					j loop3
					
				exit_loop3:
					sw $s6 0($s3)
					add $s6 $zero $zero
					j loop2
				
					
		exit_mult:
		
		lw $s0 ($sp)
		lw $s1 4($sp)
		lw $s2 8($sp)
		lw $s3 12($sp)
		lw $s4 16($sp)
		lw $s5 20($sp)
		lw $s6 24($sp)
		lw $s7 28($sp)
		addi $sp $sp 32
		jr $ra
	
	randomizer:			# creates a random number between 1-20
		randomLoop1:
			li $v0 42
			syscall
			addi $a0 $a0 1
			sw $a0 ($a2)	# stores into $a2
			addi $a2 $a2 4	
			beq $a2 $t4 exitRandomLoop1
			j randomLoop1
		exitRandomLoop1:
		jr $ra
		
		
	matrixPrinter:		# address in $a0, rows in $a1, cols in $a2
		addi $sp $sp -24
		sw $ra ($sp)
		sw $a0 4($sp)
		sw $s0 8($sp)
		sw $s1 12($sp)
		sw $s2 16($sp)
		sw $s3 20($sp)
		
		add $s0 $a0 $zero			# s0 = a0, address
		loop1Printer:
			add $s2 $zero $zero		# j = 0
			loop2Printer:
				mul $s3 $s1 $a1		# mul by rows instead of 4
				add $s3 $s3 $s2		# s3 += j
				sll $s3 $s3 2		# s3 *= 4
				add $a3 $s0 $s3		# a3 = s0[s3]	
				jal printArrElement	
				jal printTab		
				addi $s2 $s2 1		# j++
				bne $s2 $a2 loop2Printer		
			exitLoop2Printer:
			
			addi $s1 $s1 1			# i++
			beq $s1 $a1 exitLoop1Printer
			jal printNewline
			j loop1Printer
		exitLoop1Printer:
		lw $ra ($sp)
		lw $a0 4($sp)
		lw $s0 8($sp)
		lw $s1 12($sp)
		lw $s2 16($sp)
		lw $s3 20($sp)
		addi $sp $sp 24
		jr $ra
		
	printArrElement:				# elem address in $a3
		lw $a0 ($a3)
		li $v0 1
		syscall
		jr $ra
	printNewline:
		li $a0 10
		li $v0 11
		syscall
		jr $ra
	printTab:
		li $a0 9
		li $v0 11
		syscall
		jr $ra
	printString:
		li $v0 4
		syscall
		jr $ra
