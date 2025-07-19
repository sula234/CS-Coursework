#Best configuration is here :
#Number of blocks : 32
#Cache block size: 1
#The reasons for my optimization are below:
#1) In Assembly code: instead of indexing in arrays, I worked directly with addresses. Also, 
    # I removed function for swaping to delete jal and jr instructions and s registers where it is safe. 
    # Since s registers require to keep them in stack.
#2) In my opinion , the particular configuration of the Number of blocks and
#   Cache block size , which I noted above works best , because − from the formula 
#   for miss penalty, we see that by reducing the cache blocks size we will reduce our miss penalty and then reduce
#   metric score(of course our miss rate also changed, but from result we see that the change is not so big).

.data
	randoms: .space 2000	# for 10 elements
	size: .word 2000
.text
	la $a2 randoms
	li $a1 599			# upper bound for RNG (random num generator), DONT change this.
	# 10 elements
	# 20 elements
	# 50 elements
	# 100 elements 
	# 200 elements
	# 500 elements
	
	li $t4 268502992
	loopRandom:
		li $a0 0		# random i.d.
		li $v0 42		# generate random
		syscall			# random in $a0
		
		sw $a0 ($a2)		# store random to memory
		addi $a2 $a2 4		# move memory address
		beq $t4 $a2 exitLoopRandom	# stop when we reach the size
		j loopRandom
	exitLoopRandom:
	
	la $a0 randoms
	add $a1 $a0 $zero
	addi $a2 $t4 -4
	jal quicksort
	
	#la $a0 randoms
	#srl $a1 $a3 2
	#jal bubble_sort_book
		
	li $v0, 10 # Terminate program run and
	syscall # Exit

	quicksort: # a0 = arr, a1 = low, a2 = high
		addi $sp $sp -16
		sw $s1 0($sp) # for low 
		sw $s2 4($sp) # for high
		sw $s3 8($sp) 
		sw $ra 12($sp)
		
		add $s1 $a1 $zero
		add $s2 $a2 $zero
		add $s3 $a0 $zero
		
		slt $t1 $a1 $a2
		beq $zero $t1 exit_sort
		
		jal partition
		
		add $a0 $s3 $zero
		move $a1 $s1
		addi $a2 $v0 -4
		jal quicksort
		
		add $a0 $s3 $zero
		addi $a1 $v0 4
		add $a2 $s2 $zero
		jal quicksort
		
		
		
	exit_sort:
		lw $s1 0($sp)
		lw $s2 4($sp)
		lw $s3 8($sp) 
		lw $ra 12($sp)
		addi $sp $sp 16
	
		jr $ra
		
		
	partition:
	
		lw $t7 0($a2) # find pivot 
	
		add $t9 $a0 $zero # save a0 and 
	
		addi $t6 $a1 -4 # find i
		loop:
			addi $t2 $a2 -4
			bgt $a1 $t2 exit_loop
	
			lw $t2 0($a1) # find arr[j]
	
			slt $t1 $t2 $t7 # arr[j] < pivot
			beq $t1 $zero skip_swap
	
			addi $t6 $t6 4 # i++
	
			# quick swap
			lw $t0 0($t6)
 			lw $t1 0($a1)
 			sw $t1 0($t6)
 			sw $t0 0($a1)
	
		skip_swap:
			addi $a1 $a1 4
			j loop
	
		exit_loop:
		
		# i++
		addi $t6 $t6 4
		# quick swap
		lw $t0 0($t6)
 		lw $t1 0($a2)
 		sw $t1 0($t6)
 		sw $t0 0($a2) 
	
		move $v0 $t6 
	
		jr $ra
	
	
	bubble_sort_book:
		sort: addi $sp,$sp, -20 # make room on stack for 5 registers
 			sw $ra, 16($sp)# save $ra on stack
 			sw $s3,12($sp) # save $s3 on stack
 			sw $s2, 8($sp)# save $s2 on stack
 			sw $s1, 4($sp)# save $s1 on stack
 			sw $s0, 0($sp)# save $s0 on stack
 			
 			move $s2, $a0 # copy parameter $a0 into $s2 (save $a0)
 			move $s3, $a1 # copy parameter $a1 into $s3 (save $a1)
 			move $s0, $zero# i = 0
		for1tst: slt $t0, $s0, $s3 # reg $t0 = 0 if $s0 Љ $s3 (i Љ n)
			beq $t0, $zero, exit1# go to exit1 if $s0 Љ $s3 (i Љ n)
	
			addi $s1, $s0, -1# j = i – 1
		for2tst: slti $t0, $s1, 0 # reg $t0 = 1 if $s1 < 0 (j < 0)
 			bne $t0, $zero, exit2# go to exit2 if $s1 < 0 (j < 0)
 			sll $t1, $s1, 2# reg $t1 = j * 4 
 			add $t2, $s2, $t1# reg $t2 = v + (j * 4) 
 			lw $t3, 0($t2)# reg $t3 = v[j]
 			lw $t4, 4($t2)# reg $t4 = v[j + 1]
 			slt $t0, $t4, $t3 # reg $t0 = 0 if $t4 Љ $t3 
 			beq $t0, $zero, exit2# go to exit2 if $t4 Љ $t3 
 
  			move $a0, $s2 # 1st parameter of swap is v (old $a0)
 			move $a1, $s1 # 2nd parameter of swap is j 
 			jal swap # swap code shown in Figure 2.25
 			addi $s1, $s1, -1# j –= 1
 			j for2tst # jump to test of inner loop
 
 		exit2: addi $s0, $s0, 1 # i += 1
 			j for1tst # jump to test of outer loop
 		exit1: lw $s0, 0($sp) # restore $s0 from stack
 			lw $s1, 4($sp)# restore $s1 from stack
 			lw $s2, 8($sp)# restore $s2 from stack
 			lw $s3,12($sp) # restore $s3 from stack
 			lw $ra,16($sp) # restore $ra from stack
 			addi $sp,$sp, 20 # restore stack pointer
 			jr $ra
 	
 	
 	swap: sll $t1, $a1, 2 # reg $t1 = k * 4 
 		add $t1, $a0, $t1 # reg $t1 = v + (k * 4) # reg $t1 has the address of v[k]
 		lw $t0, 0($t1) # reg $t0 (temp) = v[k]
 		lw $t2, 4($t1) # reg $t2 = v[k + 1] 	  # refers to next element of v
 		sw $t2, 0($t1) # v[k] = reg $t2
 		sw $t0, 4($t1) # v[k+1] = reg $t0 (temp)
 		jr $ra # return to calling routine
 		

 		
